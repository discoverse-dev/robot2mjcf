"""Capture MuJoCo scene renderings to image files.

This module provides utilities to load MJCF files and render them to images
with different visualization settings.
"""

import argparse
import logging
from pathlib import Path
import numpy as np

try:
    import mujoco
except ImportError:
    raise ImportError("MuJoCo is required for capture functionality. Install with: pip install mujoco")

logger = logging.getLogger(__name__)


def capture_scene(
    mjcf_path: Path,
    output_path: Path,
    width: int = 2000,
    height: int = 2000,
    enable_group2: bool = True,
    enable_group3: bool = False,
) -> None:
    """
    Capture a rendered image of the MuJoCo scene.
    
    Args:
        mjcf_path: Path to the MJCF XML file
        output_path: Path to save the output image
        width: Image width in pixels
        height: Image height in pixels
        enable_group2: Whether to enable geom group 2 (visual geometries)
        enable_group3: Whether to enable geom group 3 (collision geometries)
    """
    try:
        # Load the model
        model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        data = mujoco.MjData(model)
        
        # Ensure width and height don't exceed framebuffer limits
        # MuJoCo default framebuffer is 640x480, check model settings
        max_width = getattr(model.vis.global_, 'offwidth', 640)
        max_height = getattr(model.vis.global_, 'offheight', 480)
        
        # Adjust if requested size exceeds limits
        if width > max_width or height > max_height:
            logger.warning(f"Requested size {width}x{height} exceeds framebuffer {max_width}x{max_height}, adjusting...")
            scale = min(max_width / width, max_height / height)
            width = int(width * scale)
            height = int(height * scale)
            logger.info(f"Adjusted size to {width}x{height}")
        
        # Create renderer
        renderer = mujoco.Renderer(model, height=height, width=width)
        
        # Set camera to a good viewing angle
        # Use free camera with automatic framing
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, camera)
        
        # Calculate robot bounding box and optimal camera distance
        mujoco.mj_forward(model, data)
        
        # Get geometry AABB (Axis-Aligned Bounding Box)
        aabb_center = model.geom_aabb[:, :3].astype(np.float32).copy()
        aabb_size = model.geom_aabb[:, 3:].astype(np.float32).copy()
        
        # Handle plane geometries specially
        plane_args = np.where(model.geom_type.astype(np.int32) == 0)[0]
        aabb_center[plane_args, :] = 0
        aabb_size[plane_args, :] = model.geom_size[plane_args, :]
        aabb_size[:, :] *= 0.5
        
        # Transform AABB to world coordinates
        geom_xpos = data.geom_xpos.copy()
        geom_xmat = data.geom_xmat.copy().reshape(-1, 3, 3)
        
        size_world = np.zeros_like(aabb_size)
        for i in range(geom_xmat.shape[0]):
            size_world[i] = geom_xmat[i] @ aabb_size[i].T
        
        # Calculate global AABB
        global_aabb_min = aabb_center + geom_xpos - size_world
        global_aabb_max = aabb_center + geom_xpos + size_world
        
        # Robot AABB (excluding ground plane at index 0)
        robot_aabb_min = np.min(global_aabb_min[1:], axis=0)
        robot_aabb_max = np.max(global_aabb_max[1:], axis=0)
        robot_center = 0.5 * (robot_aabb_min + robot_aabb_max)
        robot_size = (robot_aabb_max - robot_aabb_min) * 0.5
        robot_radius = np.linalg.norm(robot_size)
        
        # Calculate minimum camera distance based on FOV
        fovy_deg = model.vis.global_.fovy  # Vertical field of view in degrees
        fovy_rad = np.deg2rad(fovy_deg)
        
        # Calculate horizontal FOV based on aspect ratio
        aspect_ratio = width / height
        fovx_rad = 2 * np.arctan(np.tan(fovy_rad / 2) * aspect_ratio)
        
        # Use the smaller FOV (more conservative) for distance calculation
        min_fov_rad = min(fovy_rad, fovx_rad)
        
        # Calculate minimum distance with safety factor
        safety_factor = 1.2
        min_distance = (robot_radius / np.tan(min_fov_rad / 2)) * safety_factor
        
        # Set camera parameters
        camera.lookat = robot_center
        camera.distance = min_distance
        camera.azimuth = 150  # Rotate around z-axis
        camera.elevation = -15  # Look down at the model
        
        logger.info(f"Robot center: {robot_center}")
        logger.info(f"Robot radius: {robot_radius:.3f}")
        logger.info(f"Camera distance: {min_distance:.3f}")

        # Configure visualization options
        scene_option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(scene_option)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        
        # Configure geom groups
        # Group 0: typically static world geometry
        # Group 1: typically dynamic bodies
        # Group 2: typically visual geometries
        # Group 3: typically collision geometries
        scene_option.geomgroup[0] = 1
        scene_option.geomgroup[1] = 1
        scene_option.geomgroup[2] = 1 if enable_group2 else 0
        scene_option.geomgroup[3] = 1 if enable_group3 else 0
        
        # Update the scene
        renderer.update_scene(data, camera=camera, scene_option=scene_option)
        
        # Render RGB image
        pixels = renderer.render()
        
        # Render segmentation to create alpha channel for transparent background
        renderer.enable_segmentation_rendering()
        seg_pixels = renderer.render()[:, :, 0]  # Only take first channel
        
        # Create RGBA image with transparency based on segmentation
        # seg_pixels == 0 means background (no geometry), set alpha to 0 (transparent)
        # seg_pixels != 0 means foreground (robot geometry), set alpha to 255 (opaque)
        rgba_pixels = np.zeros((pixels.shape[0], pixels.shape[1], 4), dtype=np.uint8)
        rgba_pixels[:, :, :3] = pixels  # Copy RGB channels
        rgba_pixels[:, :, 3] = np.where(seg_pixels < 1, 0, 255)  # Set alpha based on segmentation

        logger.info(f"Background pixels: {np.sum(seg_pixels == 0)}, Foreground pixels: {np.sum(seg_pixels != 0)}")
        
        # Crop transparent borders while maintaining aspect ratio
        # Find bounding box of non-transparent pixels
        alpha_channel = rgba_pixels[:, :, 3]
        non_transparent = np.where(alpha_channel > 0)
        
        if len(non_transparent[0]) > 0:
            # Get bounding box coordinates
            y_min, y_max = non_transparent[0].min(), non_transparent[0].max()
            x_min, x_max = non_transparent[1].min(), non_transparent[1].max()
            
            # Add small padding (5% of image size or 10 pixels, whichever is smaller)
            padding = min(10, int(min(width, height) * 0.05))
            y_min = max(0, y_min - padding)
            y_max = min(rgba_pixels.shape[0], y_max + padding + 1)
            x_min = max(0, x_min - padding)
            x_max = min(rgba_pixels.shape[1], x_max + padding + 1)
            
            # Calculate crop dimensions
            crop_width = x_max - x_min
            crop_height = y_max - y_min
            
            # Calculate original aspect ratio
            original_aspect = width / height
            crop_aspect = crop_width / crop_height
            
            # Adjust crop to maintain original aspect ratio
            if crop_aspect > original_aspect:
                # Crop is wider, need to expand height
                target_height = int(crop_width / original_aspect)
                height_diff = target_height - crop_height
                y_center = (y_min + y_max) // 2
                y_min = max(0, y_center - target_height // 2)
                y_max = min(rgba_pixels.shape[0], y_min + target_height)
                # Adjust if we hit the boundary
                if y_max - y_min < target_height:
                    y_min = max(0, y_max - target_height)
            else:
                # Crop is taller, need to expand width
                target_width = int(crop_height * original_aspect)
                width_diff = target_width - crop_width
                x_center = (x_min + x_max) // 2
                x_min = max(0, x_center - target_width // 2)
                x_max = min(rgba_pixels.shape[1], x_min + target_width)
                # Adjust if we hit the boundary
                if x_max - x_min < target_width:
                    x_min = max(0, x_max - target_width)
            
            # Crop the image
            rgba_pixels = rgba_pixels[y_min:y_max, x_min:x_max]
            final_aspect = rgba_pixels.shape[1] / rgba_pixels.shape[0]
            logger.info(f"Cropped from {width}x{height} to {rgba_pixels.shape[1]}x{rgba_pixels.shape[0]}")
            logger.info(f"Original aspect ratio: {original_aspect:.3f}, Final aspect ratio: {final_aspect:.3f}")
        else:
            logger.warning("No non-transparent pixels found, skipping crop")
        
        # Resize back to original dimensions
        if rgba_pixels.shape[0] != height or rgba_pixels.shape[1] != width:
            try:
                from PIL import Image
                # Convert to PIL Image for high-quality resizing
                temp_image = Image.fromarray(rgba_pixels, mode='RGBA')
                # Use LANCZOS for high-quality downsampling
                temp_image = temp_image.resize((width, height), Image.Resampling.LANCZOS)
                rgba_pixels = np.array(temp_image)
                logger.info(f"Resized back to original dimensions: {width}x{height}")
            except ImportError:
                # Fallback: use basic numpy resize (lower quality)
                from scipy import ndimage
                zoom_factors = (height / rgba_pixels.shape[0], width / rgba_pixels.shape[1], 1)
                rgba_pixels = ndimage.zoom(rgba_pixels, zoom_factors, order=3)
                logger.info(f"Resized back to original dimensions using scipy: {width}x{height}")
        
        # Save using PIL for better format support (PNG supports transparency)
        try:
            from PIL import Image
            image = Image.fromarray(rgba_pixels, mode='RGBA')
            image.save(output_path)
            logger.info(f"Saved image with transparency to: {output_path}")
        except ImportError:
            # Fallback to matplotlib if PIL is not available
            # Note: matplotlib's imsave may not handle RGBA correctly for all formats
            import matplotlib.pyplot as plt
            plt.imsave(output_path, rgba_pixels)
            logger.info(f"Saved image to: {output_path} (using matplotlib)")
        
        # Clean up
        renderer.close()
        
    except Exception as e:
        logger.error(f"Failed to capture scene: {e}")
        raise


def capture_robot_images(mjcf_path: Path) -> None:
    """
    Capture standard robot visualization images.
    
    Creates two images:
    1. robot_name.png - Standard visualization with visual geometries
    2. collision.png - Collision geometry visualization
    
    Args:
        mjcf_path: Path to the MJCF XML file
    """
    mjcf_path = Path(mjcf_path)
    
    if not mjcf_path.exists():
        raise FileNotFoundError(f"MJCF file not found: {mjcf_path}")
    
    output_dir = mjcf_path.parent
    robot_name = mjcf_path.stem
    
    print(f"Capturing images for {robot_name}...")
    
    # Capture standard view (visual geometries)
    standard_output = output_dir / f"{robot_name}.png"
    print(f"  Rendering standard view...")
    capture_scene(
        mjcf_path=mjcf_path,
        output_path=standard_output,
        enable_group2=True,  # Enable visual geometries
        enable_group3=False,  # Disable collision geometries
    )
    print(f"  ✓ Saved: {standard_output}")
    
    # Capture collision view
    collision_output = output_dir / "collision.png"
    print(f"  Rendering collision view...")
    capture_scene(
        mjcf_path=mjcf_path,
        output_path=collision_output,
        enable_group2=False,  # Disable visual geometries
        enable_group3=True,   # Enable collision geometries
    )
    print(f"  ✓ Saved: {collision_output}")
    
    print("✅ Image capture complete!")


def main():
    """Command-line interface for capture functionality."""
    parser = argparse.ArgumentParser(
        description="Capture MuJoCo scene renderings to image files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Capture standard images for a robot
  %(prog)s robot.xml
  
  # Custom output with specific dimensions
  %(prog)s robot.xml --output custom.png --width 3840 --height 2160
  
  # Show only collision geometries
  %(prog)s robot.xml --output collision.png --no-visual --collision
        """
    )
    
    parser.add_argument(
        "mjcf_path",
        type=str,
        help="Path to the MJCF XML file"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output image path (default: robot_name.png and collision.png in same directory)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=2000,
        help="Image width in pixels (default: 2000)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=2000,
        help="Image height in pixels (default: 2000)"
    )
    parser.add_argument(
        "--no-visual",
        action="store_true",
        help="Disable visual geometries (geom group 2)"
    )
    parser.add_argument(
        "--collision",
        action="store_true",
        help="Enable collision geometries (geom group 3)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(levelname)s: %(message)s'
    )
    
    mjcf_path = Path(args.mjcf_path)
    
    if args.output:
        # Custom single output
        output_path = Path(args.output)
        capture_scene(
            mjcf_path=mjcf_path,
            output_path=output_path,
            width=args.width,
            height=args.height,
            enable_group2=not args.no_visual,
            enable_group3=args.collision,
        )
    else:
        # Standard two-image output
        capture_robot_images(mjcf_path)


if __name__ == "__main__":
    main()
