//! Minimal Vulkan presenter: uploads the frame to an image and blits it, scaled and
//! letterboxed, straight onto the swapchain. No shaders, pipelines or render passes.

#![allow(unsafe_op_in_unsafe_fn)] // this whole module is raw Vulkan calls

use crate::Result;
use ash::{khr, vk};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::Window;

/// Largest rectangle with the aspect ratio of `src` centered in `win`: (x, y, width, height).
pub fn fit(win: (u32, u32), src: (u32, u32)) -> (i32, i32, i32, i32) {
    let scale = f64::min(win.0 as f64 / src.0 as f64, win.1 as f64 / src.1 as f64);
    let w = ((src.0 as f64 * scale).round() as i32).clamp(1, win.0.max(1) as i32);
    let h = ((src.1 as f64 * scale).round() as i32).clamp(1, win.1.max(1) as i32);
    ((win.0 as i32 - w) / 2, (win.1 as i32 - h) / 2, w, h)
}

/// The remote frame on the GPU, plus the mapped staging buffer it is uploaded through.
#[derive(Clone, Copy)]
struct Source {
    image: vk::Image,
    image_memory: vk::DeviceMemory,
    buffer: vk::Buffer,
    buffer_memory: vk::DeviceMemory,
    mapped: *mut u8,
    w: u32,
    h: u32,
    initialized: bool, // false until the first full upload
}

pub struct Renderer {
    _entry: ash::Entry,
    instance: ash::Instance,
    surface_loader: khr::surface::Instance,
    surface: vk::SurfaceKHR,
    physical: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    swapchain_loader: khr::swapchain::Device,
    surface_format: vk::SurfaceFormatKHR,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    rendered: Vec<vk::Semaphore>, // one per swapchain image
    extent: vk::Extent2D,
    window_size: (u32, u32),
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    acquired: vk::Semaphore,
    fence: vk::Fence,
    source: Option<Source>,
}

const COLOR_LAYERS: vk::ImageSubresourceLayers = vk::ImageSubresourceLayers {
    aspect_mask: vk::ImageAspectFlags::COLOR,
    mip_level: 0,
    base_array_layer: 0,
    layer_count: 1,
};
const COLOR_RANGE: vk::ImageSubresourceRange = vk::ImageSubresourceRange {
    aspect_mask: vk::ImageAspectFlags::COLOR,
    base_mip_level: 0,
    level_count: 1,
    base_array_layer: 0,
    layer_count: 1,
};

impl Renderer {
    pub fn new(window: &Window) -> Result<Self> {
        let display_handle = window.display_handle()?.as_raw();
        let window_handle = window.window_handle()?.as_raw();
        let macos = cfg!(target_os = "macos"); // MoltenVK is a "portability" driver
        unsafe {
            let entry = ash::Entry::load().map_err(|e| format!("Vulkan is not available: {e}"))?;

            let mut extensions = ash_window::enumerate_required_extensions(display_handle)?.to_vec();
            let mut flags = vk::InstanceCreateFlags::empty();
            if macos {
                extensions.push(khr::portability_enumeration::NAME.as_ptr());
                flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
            }
            let app_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_0);
            let instance = entry.create_instance(
                &vk::InstanceCreateInfo::default()
                    .application_info(&app_info)
                    .enabled_extension_names(&extensions)
                    .flags(flags),
                None,
            )?;
            let surface_loader = khr::surface::Instance::new(&entry, &instance);
            let surface = ash_window::create_surface(&entry, &instance, display_handle, window_handle, None)?;

            // First GPU with a queue that can both run blits (graphics) and present to the window.
            let (physical, family) = instance
                .enumerate_physical_devices()?
                .into_iter()
                .find_map(|physical| {
                    let families = instance.get_physical_device_queue_family_properties(physical);
                    families.iter().enumerate().find_map(|(i, family)| {
                        let presents = surface_loader
                            .get_physical_device_surface_support(physical, i as u32, surface)
                            .unwrap_or(false);
                        (presents && family.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                            .then_some((physical, i as u32))
                    })
                })
                .ok_or("no Vulkan device can present to this window")?;

            let priorities = [1.0f32];
            let queues = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family)
                .queue_priorities(&priorities)];
            let mut device_extensions = vec![khr::swapchain::NAME.as_ptr()];
            if macos {
                device_extensions.push(khr::portability_subset::NAME.as_ptr());
            }
            let device = instance.create_device(
                physical,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queues)
                    .enabled_extension_names(&device_extensions),
                None,
            )?;
            let queue = device.get_device_queue(family, 0);
            let swapchain_loader = khr::swapchain::Device::new(&instance, &device);

            let pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?;
            let cmd = device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];
            let acquired = device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
            let fence = device.create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )?;

            // Prefer a plain (non-sRGB) BGRA surface: the frames are already display-encoded.
            let formats = surface_loader.get_physical_device_surface_formats(physical, surface)?;
            let surface_format = *formats
                .iter()
                .find(|f| f.format == vk::Format::B8G8R8A8_UNORM)
                .or_else(|| formats.iter().find(|f| f.format == vk::Format::R8G8B8A8_UNORM))
                .or(formats.first())
                .ok_or("the window surface has no formats")?;

            let size = window.inner_size();
            let mut renderer = Renderer {
                _entry: entry,
                instance,
                surface_loader,
                surface,
                physical,
                device,
                queue,
                swapchain_loader,
                surface_format,
                swapchain: vk::SwapchainKHR::null(),
                images: Vec::new(),
                rendered: Vec::new(),
                extent: vk::Extent2D::default(),
                window_size: (size.width, size.height),
                pool,
                cmd,
                acquired,
                fence,
                source: None,
            };
            renderer.create_swapchain()?;
            Ok(renderer)
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        self.window_size = (width, height);
        unsafe { self.create_swapchain() }
    }

    /// (Re)creates the swapchain for the current window size. While the window is
    /// minimized the extent is left at zero and `render` does nothing.
    unsafe fn create_swapchain(&mut self) -> Result<()> {
        let caps = self
            .surface_loader
            .get_physical_device_surface_capabilities(self.physical, self.surface)?;
        let extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            vk::Extent2D {
                width: self.window_size.0.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                height: self.window_size.1.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            }
        };
        self.extent = extent;
        if extent.width == 0 || extent.height == 0 {
            return Ok(());
        }
        if !caps.supported_usage_flags.contains(vk::ImageUsageFlags::TRANSFER_DST) {
            return Err("this Vulkan driver cannot blit to the window".into());
        }

        // Lowest latency first: never wait for vblank if the driver lets us.
        let modes = self
            .surface_loader
            .get_physical_device_surface_present_modes(self.physical, self.surface)?;
        let present_mode = [vk::PresentModeKHR::MAILBOX, vk::PresentModeKHR::IMMEDIATE]
            .into_iter()
            .find(|mode| modes.contains(mode))
            .unwrap_or(vk::PresentModeKHR::FIFO);
        let composite_alpha = [vk::CompositeAlphaFlagsKHR::OPAQUE, vk::CompositeAlphaFlagsKHR::INHERIT]
            .into_iter()
            .find(|&alpha| caps.supported_composite_alpha.contains(alpha))
            .unwrap_or(vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED);
        let mut image_count = caps.min_image_count + 1;
        if caps.max_image_count != 0 {
            image_count = image_count.min(caps.max_image_count);
        }

        self.device.device_wait_idle()?;
        let old = self.swapchain;
        self.swapchain = self.swapchain_loader.create_swapchain(
            &vk::SwapchainCreateInfoKHR::default()
                .surface(self.surface)
                .min_image_count(image_count)
                .image_format(self.surface_format.format)
                .image_color_space(self.surface_format.color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(caps.current_transform)
                .composite_alpha(composite_alpha)
                .present_mode(present_mode)
                .clipped(true)
                .old_swapchain(old),
            None,
        )?;
        self.swapchain_loader.destroy_swapchain(old, None);
        self.images = self.swapchain_loader.get_swapchain_images(self.swapchain)?;
        while self.rendered.len() < self.images.len() {
            let semaphore = self.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
            self.rendered.push(semaphore);
        }
        Ok(())
    }

    /// Makes sure the source image and staging buffer match the remote resolution.
    unsafe fn ensure_source(&mut self, w: u32, h: u32) -> Result<Source> {
        if let Some(source) = self.source.filter(|s| (s.w, s.h) == (w, h)) {
            return Ok(source);
        }
        self.destroy_source();

        // Match the swapchain's sRGB-ness so the blit never converts colors.
        let srgb = matches!(
            self.surface_format.format,
            vk::Format::B8G8R8A8_SRGB | vk::Format::R8G8B8A8_SRGB | vk::Format::A8B8G8R8_SRGB_PACK32
        );
        let format = if srgb { vk::Format::B8G8R8A8_SRGB } else { vk::Format::B8G8R8A8_UNORM };
        let image = self.device.create_image(
            &vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D { width: w, height: h, depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST),
            None,
        )?;
        let image_memory = self.allocate(
            self.device.get_image_memory_requirements(image),
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        self.device.bind_image_memory(image, image_memory, 0)?;

        let size = w as u64 * h as u64 * 4;
        let buffer = self.device.create_buffer(
            &vk::BufferCreateInfo::default().size(size).usage(vk::BufferUsageFlags::TRANSFER_SRC),
            None,
        )?;
        let buffer_memory = self.allocate(
            self.device.get_buffer_memory_requirements(buffer),
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        self.device.bind_buffer_memory(buffer, buffer_memory, 0)?;
        let mapped = self.device.map_memory(buffer_memory, 0, size, vk::MemoryMapFlags::empty())? as *mut u8;

        let source = Source { image, image_memory, buffer, buffer_memory, mapped, w, h, initialized: false };
        self.source = Some(source);
        Ok(source)
    }

    unsafe fn allocate(
        &self,
        requirements: vk::MemoryRequirements,
        flags: vk::MemoryPropertyFlags,
    ) -> Result<vk::DeviceMemory> {
        let properties = self.instance.get_physical_device_memory_properties(self.physical);
        let index = (0..properties.memory_type_count)
            .find(|&i| {
                requirements.memory_type_bits & (1 << i) != 0
                    && properties.memory_types[i as usize].property_flags.contains(flags)
            })
            .ok_or("no suitable GPU memory type")?;
        Ok(self.device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(index),
            None,
        )?)
    }

    unsafe fn destroy_source(&mut self) {
        if let Some(source) = self.source.take() {
            let _ = self.device.device_wait_idle();
            self.device.destroy_image(source.image, None);
            self.device.free_memory(source.image_memory, None);
            self.device.destroy_buffer(source.buffer, None);
            self.device.free_memory(source.buffer_memory, None); // also unmaps it
        }
    }

    unsafe fn transition(&self, image: vk::Image, from: vk::ImageLayout, to: vk::ImageLayout) {
        let access = |layout| match layout {
            vk::ImageLayout::TRANSFER_DST_OPTIMAL => vk::AccessFlags::TRANSFER_WRITE,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL => vk::AccessFlags::TRANSFER_READ,
            _ => vk::AccessFlags::empty(),
        };
        let barrier = vk::ImageMemoryBarrier::default()
            .image(image)
            .old_layout(from)
            .new_layout(to)
            .src_access_mask(access(from))
            .dst_access_mask(access(to))
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(COLOR_RANGE);
        self.device.cmd_pipeline_barrier(
            self.cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }

    /// Presents the `w` x `h` BGRA frame scaled to fit the window. Only the `dirty` rows
    /// [start, end) are uploaded; the GPU copy of the rest is reused. Returns false if
    /// nothing was drawn (e.g. the window is minimized).
    pub fn render(&mut self, bgra: &[u8], w: u32, h: u32, dirty: Option<(u32, u32)>) -> Result<bool> {
        if self.extent.width == 0 || self.extent.height == 0 || bgra.len() != w as usize * h as usize * 4 {
            return Ok(false);
        }
        unsafe {
            // The previous frame must be off the GPU before its staging buffer is overwritten.
            self.device.wait_for_fences(&[self.fence], true, u64::MAX)?;
            let source = self.ensure_source(w, h)?;
            let dirty = if source.initialized { dirty } else { Some((0, h)) };
            let row = w as usize * 4;
            if let Some((y0, y1)) = dirty {
                let (start, end) = (y0 as usize * row, y1 as usize * row);
                std::ptr::copy_nonoverlapping(bgra[start..end].as_ptr(), source.mapped.add(start), end - start);
            }

            let mut retried = false;
            let index = loop {
                match self
                    .swapchain_loader
                    .acquire_next_image(self.swapchain, u64::MAX, self.acquired, vk::Fence::null())
                {
                    Ok((index, _)) => break index,
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) if !retried => {
                        retried = true;
                        self.create_swapchain()?;
                        if self.extent.width == 0 || self.extent.height == 0 {
                            return Ok(false);
                        }
                    }
                    Err(e) => return Err(e.into()),
                }
            };
            let target = self.images[index as usize];
            let (device, cmd) = (&self.device, self.cmd);

            device.reset_fences(&[self.fence])?;
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            // Staging buffer -> source image, changed rows only.
            if let Some((y0, y1)) = dirty {
                let current = if source.initialized {
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL
                } else {
                    vk::ImageLayout::UNDEFINED
                };
                self.transition(source.image, current, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
                device.cmd_copy_buffer_to_image(
                    cmd,
                    source.buffer,
                    source.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[vk::BufferImageCopy::default()
                        .buffer_offset(y0 as u64 * row as u64)
                        .image_subresource(COLOR_LAYERS)
                        .image_offset(vk::Offset3D { x: 0, y: y0 as i32, z: 0 })
                        .image_extent(vk::Extent3D { width: w, height: y1 - y0, depth: 1 })],
                );
                self.transition(source.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
            }

            // Source image -> swapchain image, scaled, with black bars around it.
            self.transition(target, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
            device.cmd_clear_color_image(
                cmd,
                target,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] },
                &[COLOR_RANGE],
            );
            let (x, y, fit_w, fit_h) = fit((self.extent.width, self.extent.height), (w, h));
            let blit = vk::ImageBlit::default()
                .src_subresource(COLOR_LAYERS)
                .src_offsets([vk::Offset3D::default(), vk::Offset3D { x: w as i32, y: h as i32, z: 1 }])
                .dst_subresource(COLOR_LAYERS)
                .dst_offsets([vk::Offset3D { x, y, z: 0 }, vk::Offset3D { x: x + fit_w, y: y + fit_h, z: 1 }]);
            device.cmd_blit_image(
                cmd,
                source.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                target,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[blit],
                vk::Filter::LINEAR,
            );
            self.transition(target, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR);
            device.end_command_buffer(cmd)?;

            let wait = [self.acquired];
            let stages = [vk::PipelineStageFlags::TRANSFER];
            let cmds = [cmd];
            let signal = [self.rendered[index as usize]];
            device.queue_submit(
                self.queue,
                &[vk::SubmitInfo::default()
                    .wait_semaphores(&wait)
                    .wait_dst_stage_mask(&stages)
                    .command_buffers(&cmds)
                    .signal_semaphores(&signal)],
                self.fence,
            )?;
            if let Some(source) = &mut self.source {
                source.initialized = true;
            }

            let swapchains = [self.swapchain];
            let indices = [index];
            let present = vk::PresentInfoKHR::default()
                .wait_semaphores(&signal)
                .swapchains(&swapchains)
                .image_indices(&indices);
            match self.swapchain_loader.queue_present(self.queue, &present) {
                Ok(_) => Ok(true),
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.create_swapchain().map(|()| true),
                Err(e) => Err(e.into()),
            }
        }
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.destroy_source();
            for &semaphore in &self.rendered {
                self.device.destroy_semaphore(semaphore, None);
            }
            self.device.destroy_semaphore(self.acquired, None);
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.pool, None);
            self.swapchain_loader.destroy_swapchain(self.swapchain, None);
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}
