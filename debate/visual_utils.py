"""
Utility functions for visualization
Contains all functions related to image processing, annotation, and visualization
"""

import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
import warnings
import logging
import numpy as np
from typing import List, Tuple, Optional, Union, Dict
import json
import re
import cv2

# Suppress matplotlib font manager warnings (including non-existent fonts like SimHei)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*findfont.*")
warnings.filterwarnings("ignore", message=".*Font family.*not found.*")

def zoom_in_region(image: Image.Image, 
                   bbox: List[float], 
                   output_size: Optional[Tuple[int, int]] = None,
                   padding: float = 0.0) -> Image.Image:
    """
    Crop and scale image region based on normalized coordinates
    
    Args:
        image: Input PIL Image object
        bbox: Normalized bounding box [x, y, w, h], all values in [0, 1] range
        output_size: Output image size (width, height), None to keep original crop size
        padding: Bounding box expansion ratio (0.0-1.0), 0 means no expansion
    
    Returns:
        Cropped and possibly scaled PIL Image object
    """
    if len(bbox) != 4:
        raise ValueError("bbox must contain 4 values: [x, y, w, h]")
    
    x_norm, y_norm, w_norm, h_norm = bbox
    
    if not all(0 <= val <= 1 for val in bbox):
        raise ValueError("All bbox values must be in [0, 1] range")
    
    if w_norm <= 0 or h_norm <= 0:
        raise ValueError("Width and height must be greater than 0")
    
    img_width, img_height = image.size
    
    x_pixel = x_norm * img_width
    y_pixel = y_norm * img_height
    w_pixel = w_norm * img_width
    h_pixel = h_norm * img_height
    
    if padding > 0:
        padding_x = w_pixel * padding
        padding_y = h_pixel * padding
        
        x_pixel = max(0, x_pixel - padding_x)
        y_pixel = max(0, y_pixel - padding_y)
        w_pixel = min(img_width - x_pixel, w_pixel + 2 * padding_x)
        h_pixel = min(img_height - y_pixel, h_pixel + 2 * padding_y)
    
    x_pixel = max(0, min(x_pixel, img_width))
    y_pixel = max(0, min(y_pixel, img_height))
    x2_pixel = min(img_width, x_pixel + w_pixel)
    y2_pixel = min(img_height, y_pixel + h_pixel)
    
    actual_w = x2_pixel - x_pixel
    actual_h = y2_pixel - y_pixel
    
    if actual_w <= 0 or actual_h <= 0:
        raise ValueError("Invalid crop region: computed region is empty")
    
    crop_box = (int(x_pixel), int(y_pixel), int(x2_pixel), int(y2_pixel))
    cropped_image = image.crop(crop_box)
    
    if output_size is not None:
        output_width, output_height = output_size
        if output_width <= 0 or output_height <= 0:
            raise ValueError("Output size must be greater than 0")
        cropped_image = cropped_image.resize((output_width, output_height), 
                                           Image.Resampling.LANCZOS)
    
    return cropped_image


def convert_normalized_bbox_to_pixels(bbox, img_width, img_height):
    """Convert normalized bbox [x, y, w, h] to pixel coordinates [x1, y1, x2, y2]."""
    if len(bbox) != 4:
        raise ValueError(f"Expected bbox with 4 values, got {len(bbox)}")
    
    x, y, w, h = bbox
    
    x1 = x * img_width
    y1 = y * img_height
    x2 = (x + w) * img_width
    y2 = (y + h) * img_height
    
    x1 = max(0, min(x1, img_width))
    y1 = max(0, min(y1, img_height))
    x2 = max(0, min(x2, img_width))
    y2 = max(0, min(y2, img_height))
    
    if x2 <= x1:
        x2 = min(x1 + 1, img_width)
    if y2 <= y1:
        y2 = min(y1 + 1, img_height)
        
    return [x1, y1, x2, y2]

def find_non_overlapping_position(bbox, label, text_positions, width, height, font_size=12, is_point=False):
    """Find a position for text that doesn't overlap with existing text or go outside image bounds."""
    text_width = len(label) * font_size * 0.6
    text_height = font_size + 8
    
    if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        if is_point:
            center_x = width // 2
            center_y = height // 2
            bbox = [center_x - 50, center_y - 50, center_x + 50, center_y + 50]
        else:
            bbox = [0, 0, min(100, width), min(100, height)]
    
    candidate_positions = [
        (bbox[0], bbox[1] - 20),
        (bbox[0], bbox[3] + 10),
        (bbox[2] + 10, bbox[1]),
        (bbox[2] + 10, bbox[3] - text_height),
        (bbox[0] - text_width - 10, bbox[1]),
        (bbox[0] - text_width - 10, bbox[3] - text_height),
        (bbox[0] + (bbox[2] - bbox[0] - text_width) / 2, bbox[1] - 20),
        (bbox[0] + (bbox[2] - bbox[0] - text_width) / 2, bbox[3] + 10),
        (bbox[0] - text_width - 20, bbox[1] + (bbox[3] - bbox[1]) / 2),
        (bbox[2] + 20, bbox[1] + (bbox[3] - bbox[1]) / 2),
    ]
    
    for i, (x, y) in enumerate(candidate_positions):
        if (x >= 0 and y >= text_height and 
            x + text_width <= width and y + text_height <= height):
            
            overlap = False
            for existing_x, existing_y, existing_w, existing_h in text_positions:
                if not (x + text_width + 5 < existing_x or x > existing_x + existing_w + 5 or
                        y + text_height + 5 < existing_y or y > existing_y + existing_h + 5):
                    overlap = True
                    break
            
            if not overlap:
                return x, y, text_width, text_height
    
    base_x, base_y = bbox[0], bbox[1] - 20
    offset_y = len(text_positions) * 25
    offset_x = (len(text_positions) % 3) * 100
    
    final_x = max(10, min(base_x + offset_x, width - text_width - 10))
    final_y = max(text_height + 5, min(base_y - offset_y, height - text_height - 5))
    
    return final_x, final_y, text_width, text_height

def draw_text_with_font(ax, text_x, text_y, label, color, chinese_font, font_size=12):
    """Draw text with Chinese font support."""
    if chinese_font:
        plt.text(text_x, text_y, label, color=color, fontsize=font_size, fontweight='bold',
               fontproperties=chinese_font,
               bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor=color, linewidth=2))
    else:
        plt.text(text_x, text_y, label, color=color, fontsize=font_size, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor=color, linewidth=2))

def draw_point_annotation(ax, item, i, color, width, height, text_positions, shown_labels, chinese_font):
    """Draw point annotation on the plot."""
    point = item['point_2d']
    label = item['label']
    
    if not point or len(point) != 2:
        return False
    
    try:
        pixel_x = point[0] * width
        pixel_y = point[1] * height
        
        pixel_x = max(0, min(pixel_x, width))
        pixel_y = max(0, min(pixel_y, height))
    except Exception as e:
        return False
    
    circle = patches.Circle((pixel_x, pixel_y), radius=8, 
                          linewidth=3, edgecolor=color, facecolor=color, alpha=0.7)
    ax.add_patch(circle)
    
    ax.text(pixel_x, pixel_y, str(i+1), 
           ha='center', va='center', fontsize=12, fontweight='bold',
           color='white', bbox=dict(boxstyle="circle,pad=0.2", facecolor=color, edgecolor=color))
    
    if label not in shown_labels:
        shown_labels.add(label)
        
        point_bbox = [pixel_x-10, pixel_y-10, pixel_x+10, pixel_y+10]
        text_x, text_y, text_w, text_h = find_non_overlapping_position(point_bbox, label, text_positions, width, height, font_size=12, is_point=True)
        text_positions.append((text_x, text_y, text_w, text_h))
        
        line_end_x = text_x + text_w / 2
        line_end_y = text_y + text_h / 2
        
        ax.plot([pixel_x, line_end_x], [pixel_y, line_end_y], 
               color=color, linestyle='--', alpha=0.6, linewidth=2)
        
        draw_text_with_font(ax, text_x, text_y, label, color, chinese_font)
    
    return True

def draw_bbox_annotation(ax, item, i, color, width, height, text_positions, shown_labels, chinese_font):
    """Draw bounding box annotation on the plot."""
    bbox = item['bbox_2d']
    label = item['label']
    
    if not bbox or len(bbox) != 4:
        return False
    
    try:
        pixel_bbox = convert_normalized_bbox_to_pixels(bbox, width, height)
    except Exception as e:
        return False
    
    rect = patches.Rectangle((pixel_bbox[0], pixel_bbox[1]), 
                           pixel_bbox[2] - pixel_bbox[0], pixel_bbox[3] - pixel_bbox[1], 
                           linewidth=3, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    
    box_center_x = pixel_bbox[0] + (pixel_bbox[2] - pixel_bbox[0]) / 2
    box_center_y = pixel_bbox[1] + (pixel_bbox[3] - pixel_bbox[1]) / 2
    ax.text(box_center_x, box_center_y, str(i+1), 
           ha='center', va='center', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle="circle,pad=0.3", facecolor='white', edgecolor=color))
    
    if label not in shown_labels:
        shown_labels.add(label)
        
        text_x, text_y, text_w, text_h = find_non_overlapping_position(pixel_bbox, label, text_positions, width, height, font_size=12)
        text_positions.append((text_x, text_y, text_w, text_h))
        
        line_start_x = pixel_bbox[0] + (pixel_bbox[2] - pixel_bbox[0]) / 2
        line_start_y = pixel_bbox[1] + (pixel_bbox[3] - pixel_bbox[1]) / 2
        line_end_x = text_x + text_w / 2
        line_end_y = text_y + text_h / 2
        
        ax.plot([line_start_x, line_end_x], [line_start_y, line_end_y], 
               color=color, linestyle='--', alpha=0.6, linewidth=2)
        
        draw_text_with_font(ax, text_x, text_y, label, color, chinese_font)
    
    return True

def draw_line_annotation(ax, item, i, color, width, height, text_positions, shown_labels, chinese_font):
    """Draw line annotation on the plot."""
    line = item['line_2d']
    label = item['label']
    
    if not line or len(line) != 4:
        return False
    
    try:
        x1, y1, x2, y2 = line
        
        pixel_x1 = x1 * width
        pixel_y1 = y1 * height
        pixel_x2 = x2 * width
        pixel_y2 = y2 * height
        
        pixel_x1 = max(0, min(pixel_x1, width))
        pixel_y1 = max(0, min(pixel_y1, height))
        pixel_x2 = max(0, min(pixel_x2, width))
        pixel_y2 = max(0, min(pixel_y2, height))
    except Exception as e:
        return False
    
    ax.plot([pixel_x1, pixel_x2], [pixel_y1, pixel_y2], 
           color=color, linewidth=4, alpha=0.8)
    
    ax.annotate('', xy=(pixel_x2, pixel_y2), xytext=(pixel_x1, pixel_y1),
               arrowprops=dict(arrowstyle='->', color=color, lw=3))
    
    mid_x = (pixel_x1 + pixel_x2) / 2
    mid_y = (pixel_y1 + pixel_y2) / 2
    ax.text(mid_x, mid_y, str(i+1), 
           ha='center', va='center', fontsize=12, fontweight='bold',
           color='white', bbox=dict(boxstyle="circle,pad=0.2", facecolor=color, edgecolor=color))
    
    if label not in shown_labels:
        shown_labels.add(label)
        
        line_bbox = [mid_x-20, mid_y-20, mid_x+20, mid_y+20]
        text_x, text_y, text_w, text_h = find_non_overlapping_position(line_bbox, label, text_positions, width, height, font_size=12)
        text_positions.append((text_x, text_y, text_w, text_h))
        
        line_end_x = text_x + text_w / 2
        line_end_y = text_y + text_h / 2
        
        ax.plot([mid_x, line_end_x], [mid_y, line_end_y], 
               color=color, linestyle='--', alpha=0.6, linewidth=2)
        
        draw_text_with_font(ax, text_x, text_y, label, color, chinese_font)
    
    return True

def process_zoom_operations(image, bbox_data):
    """
    Process zoom operations, extract zoom_in_2d requests from bbox_data and apply them
    Note: Only process the first zoom_in_2d request, ignore subsequent zoom requests
    output_size and padding parameters are hardcoded to fixed values, not read from input
    
    Args:
        image: Input PIL Image object
        bbox_data: List of data containing zoom_in_2d field
    
    Returns:
        Single zoom image (PIL Image object), or None if no zoom request
    """
    # Fixed parameters: do not expand bounding box, keep original crop size
    OUTPUT_SIZE = None  # None means keep original crop size, no scaling
    PADDING = 0.0  # Do not expand bounding box
    
    for item in bbox_data:
        if not isinstance(item, dict):
            continue
        
        # Check if zoom_in_2d field exists (only process the first one)
        if 'zoom_in_2d' in item:
            zoom_bbox = item['zoom_in_2d']
            label = item.get('label', 'Zoom Region')
            
            # Skip invalid zoom data
            if not zoom_bbox or len(zoom_bbox) != 4:
                continue
            
            try:
                # Use the original image for zooming (not the annotated one)
                # Use fixed output_size and padding parameters
                zoomed_region = zoom_in_region(image, zoom_bbox, OUTPUT_SIZE, PADDING)
                
                # Add label (optional)
                if label:
                    try:
                        draw = ImageDraw.Draw(zoomed_region)
                        try:
                            font = ImageFont.truetype("Arial.ttf", 16)
                        except:
                            font = ImageFont.load_default()
                        
                        # Add label at the top of the image
                        text_bbox = draw.textbbox((0, 0), label, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        
                        title_height = text_height + 10
                        title_bg = Image.new('RGB', (zoomed_region.width, title_height), 'white')
                        title_draw = ImageDraw.Draw(title_bg)
                        title_x = (zoomed_region.width - text_width) // 2
                        title_draw.text((title_x, 5), label, fill='black', font=font)
                        
                        final_image = Image.new('RGB', (zoomed_region.width, zoomed_region.height + title_height))
                        final_image.paste(title_bg, (0, 0))
                        final_image.paste(zoomed_region, (0, title_height))
                        zoomed_region = final_image
                    except:
                        # If adding label fails, continue using original zoom image
                        pass
                
                # Only process the first zoom request, return immediately after finding it
                return zoomed_region
            except Exception as e:
                print(f"Warning: Zoom operation failed: {e}")
                # Return None even on failure, do not process subsequent zoom requests
                return None
    
    return None

# Cache for fonts to avoid repeated loading
_font_cache = {}

def _get_font(size=14):
    """Get a font object, using cache to avoid repeated loading."""
    cache_key = size
    if cache_key not in _font_cache:
        try:
            # Try to load a default font
            _font_cache[cache_key] = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
        except:
            try:
                _font_cache[cache_key] = ImageFont.load_default()
            except:
                _font_cache[cache_key] = None
    return _font_cache.get(cache_key)

def create_annotated_image(image, bbox_data, use_pil_draw=True):
    """
    Create the annotated image with all non-zoom annotations.
    
    Args:
        image: PIL Image object
        bbox_data: List of annotation dicts
        use_pil_draw: If True, use PIL ImageDraw (faster). If False, use matplotlib (slower but more features)
    
    Returns:
        PIL Image with annotations
    """
    try:
        width, height = image.size
    except Exception as e:
        return image
    
    # If no bbox_data or empty, return original image
    if not bbox_data:
        return image.copy()
    
    # Use optimized PIL-based drawing if enabled
    if use_pil_draw:
        return _create_annotated_image_pil(image, bbox_data, width, height)
    
    # Fallback to matplotlib (original implementation)
    return _create_annotated_image_matplotlib(image, bbox_data, width, height)

def _create_annotated_image_pil(image, bbox_data, width, height):
    """Optimized version using PIL ImageDraw - much faster than matplotlib, with same features."""
    # Create a copy of the image to draw on
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    # Get font once
    font = _get_font(14)
    font_small = _get_font(12)
    
    shown_labels = set()
    text_positions = []
    
    # Color mapping (PIL uses RGB tuples)
    colors = [
        (255, 0, 0),      # red
        (0, 0, 255),      # blue
        (0, 255, 0),      # green
        (255, 165, 0),    # orange
        (128, 0, 128),    # purple
        (165, 42, 42),    # brown
        (255, 192, 203),  # pink
        (128, 128, 128),  # gray
        (128, 128, 0),   # olive
        (0, 255, 255),   # cyan
    ]
    
    import math
    
    for i, item in enumerate(bbox_data):
        if not isinstance(item, dict) or 'label' not in item or 'zoom_in_2d' in item or 'depth' in item:
            continue
        
        color = colors[i % len(colors)]
        label = item.get('label', '')
        
        # Draw bbox
        if 'bbox_2d' in item:
            bbox = item['bbox_2d']
            if bbox and len(bbox) == 4:
                try:
                    pixel_bbox = convert_normalized_bbox_to_pixels(bbox, width, height)
                    x1, y1, x2, y2 = pixel_bbox
                    # Draw rectangle
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    # Draw number in center
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    # Draw circle background for number
                    circle_radius = 12
                    draw.ellipse([center_x - circle_radius, center_y - circle_radius,
                                 center_x + circle_radius, center_y + circle_radius],
                                fill='white', outline=color, width=2)
                    # Draw number
                    num_text = str(i + 1)
                    if font_small:
                        bbox_text = draw.textbbox((0, 0), num_text, font=font_small)
                        text_width = bbox_text[2] - bbox_text[0]
                        text_height = bbox_text[3] - bbox_text[1]
                        draw.text((center_x - text_width / 2, center_y - text_height / 2),
                                 num_text, fill=color, font=font_small)
                    else:
                        draw.text((center_x - 5, center_y - 7), num_text, fill=color)
                    
                    # Draw label text with connecting line (if not shown before)
                    if label and label not in shown_labels:
                        shown_labels.add(label)
                        text_x, text_y, text_w, text_h = find_non_overlapping_position(
                            pixel_bbox, label, text_positions, width, height, font_size=12)
                        text_positions.append((text_x, text_y, text_w, text_h))
                        
                        # Draw dashed line from bbox center to text
                        line_end_x = text_x + text_w / 2
                        line_end_y = text_y + text_h / 2
                        _draw_dashed_line(draw, center_x, center_y, line_end_x, line_end_y, color, width=2)
                        
                        # Draw text with background
                        _draw_text_with_background(draw, text_x, text_y, label, color, font)
                except:
                    continue
        
        # Draw point
        elif 'point_2d' in item:
            point = item['point_2d']
            if point and len(point) == 2:
                try:
                    pixel_x = point[0] * width
                    pixel_y = point[1] * height
                    pixel_x = max(0, min(pixel_x, width))
                    pixel_y = max(0, min(pixel_y, height))
                    
                    # Draw circle
                    circle_radius = 8
                    draw.ellipse([pixel_x - circle_radius, pixel_y - circle_radius,
                                 pixel_x + circle_radius, pixel_y + circle_radius],
                                fill=color, outline=color, width=3)
                    # Draw number
                    num_text = str(i + 1)
                    if font_small:
                        bbox_text = draw.textbbox((0, 0), num_text, font=font_small)
                        text_width = bbox_text[2] - bbox_text[0]
                        text_height = bbox_text[3] - bbox_text[1]
                        draw.text((pixel_x - text_width / 2, pixel_y - text_height / 2),
                                 num_text, fill='white', font=font_small)
                    
                    # Draw label text with connecting line (if not shown before)
                    if label and label not in shown_labels:
                        shown_labels.add(label)
                        point_bbox = [pixel_x-10, pixel_y-10, pixel_x+10, pixel_y+10]
                        text_x, text_y, text_w, text_h = find_non_overlapping_position(
                            point_bbox, label, text_positions, width, height, font_size=12, is_point=True)
                        text_positions.append((text_x, text_y, text_w, text_h))
                        
                        # Draw dashed line from point to text
                        line_end_x = text_x + text_w / 2
                        line_end_y = text_y + text_h / 2
                        _draw_dashed_line(draw, pixel_x, pixel_y, line_end_x, line_end_y, color, width=2)
                        
                        # Draw text with background
                        _draw_text_with_background(draw, text_x, text_y, label, color, font)
                except:
                    continue
        
        # Draw line
        elif 'line_2d' in item:
            line = item['line_2d']
            if line and len(line) == 4:
                try:
                    x1 = line[0] * width
                    y1 = line[1] * height
                    x2 = line[2] * width
                    y2 = line[3] * height
                    x1 = max(0, min(x1, width))
                    y1 = max(0, min(y1, height))
                    x2 = max(0, min(x2, width))
                    y2 = max(0, min(y2, height))
                    
                    # Draw line
                    draw.line([x1, y1, x2, y2], fill=color, width=4)
                    # Draw arrow head
                    angle = math.atan2(y2 - y1, x2 - x1)
                    arrow_size = 10
                    arrow_x1 = x2 - arrow_size * math.cos(angle - math.pi / 6)
                    arrow_y1 = y2 - arrow_size * math.sin(angle - math.pi / 6)
                    arrow_x2 = x2 - arrow_size * math.cos(angle + math.pi / 6)
                    arrow_y2 = y2 - arrow_size * math.sin(angle + math.pi / 6)
                    draw.polygon([x2, y2, arrow_x1, arrow_y1, arrow_x2, arrow_y2], fill=color)
                    
                    # Draw number at midpoint
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    num_text = str(i + 1)
                    circle_radius = 10
                    draw.ellipse([mid_x - circle_radius, mid_y - circle_radius,
                                 mid_x + circle_radius, mid_y + circle_radius],
                                fill=color, outline=color, width=2)
                    if font_small:
                        bbox_text = draw.textbbox((0, 0), num_text, font=font_small)
                        text_width = bbox_text[2] - bbox_text[0]
                        text_height = bbox_text[3] - bbox_text[1]
                        draw.text((mid_x - text_width / 2, mid_y - text_height / 2),
                                 num_text, fill='white', font=font_small)
                    
                    # Draw label text with connecting line (if not shown before)
                    if label and label not in shown_labels:
                        shown_labels.add(label)
                        line_bbox = [mid_x-20, mid_y-20, mid_x+20, mid_y+20]
                        text_x, text_y, text_w, text_h = find_non_overlapping_position(
                            line_bbox, label, text_positions, width, height, font_size=12)
                        text_positions.append((text_x, text_y, text_w, text_h))
                        
                        # Draw dashed line from line midpoint to text
                        line_end_x = text_x + text_w / 2
                        line_end_y = text_y + text_h / 2
                        _draw_dashed_line(draw, mid_x, mid_y, line_end_x, line_end_y, color, width=2)
                        
                        # Draw text with background
                        _draw_text_with_background(draw, text_x, text_y, label, color, font)
                except:
                    continue
    
    return img

def _draw_dashed_line(draw, x1, y1, x2, y2, color, width=2, dash_length=5):
    """Draw a dashed line using PIL."""
    import math
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if distance == 0:
        return
    
    num_dashes = int(distance / (dash_length * 2))
    if num_dashes == 0:
        num_dashes = 1
    
    dx = (x2 - x1) / (num_dashes * 2)
    dy = (y2 - y1) / (num_dashes * 2)
    
    for i in range(num_dashes):
        start_x = x1 + i * 2 * dx
        start_y = y1 + i * 2 * dy
        end_x = start_x + dx
        end_y = start_y + dy
        draw.line([start_x, start_y, end_x, end_y], fill=color, width=width)

def _draw_text_with_background(draw, x, y, text, color, font):
    """Draw text with rounded rectangle background."""
    if font:
        bbox = draw.textbbox((x, y), text, font=font)
    else:
        # Estimate text size
        text_width = len(text) * 7
        text_height = 14
        bbox = (x, y, x + text_width, y + text_height)
    
    # Add padding
    padding = 4
    bg_bbox = [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding]
    
    # Draw rounded rectangle background (approximate with rectangle)
    draw.rectangle(bg_bbox, fill='white', outline=color, width=2)
    
    # Draw text
    if font:
        draw.text((x, y), text, fill=color, font=font)
    else:
        draw.text((x, y), text, fill=color)

def _create_annotated_image_matplotlib(image, bbox_data, width, height):
    """Original matplotlib-based implementation (slower but more features)."""
    chinese_font = None
    # # Use Linux Chinese fonts (known to be available)
    # chinese_font = None
    # try:
    #     # Linux Chinese fonts in order of preference
    #     chinese_fonts = [
    #         'SimHei', 'Arial Unicode MS', 'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
    #         'Noto Sans CJK SC', 'Noto Sans CJK', 'Source Han Sans CN', 'Source Han Sans SC', 'DejaVu Sans'
    #     ]
        
    #     for font_name in chinese_fonts:
    #         try:
    #             chinese_font = fm.FontProperties(family=font_name)
    #             break
    #         except:
    #             continue
    # except:
    #     chinese_font = None
    try:
        macos_chinese_fonts = [
            'Hiragino Sans GB',
            'STHeiti',
            'Arial Unicode MS',
            'Helvetica',
        ]
        
        # Suppress warnings during font lookup (including non-existent fonts like SimHei)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*findfont.*")
            warnings.filterwarnings("ignore", message=".*Font family.*not found.*")
            for font_name in macos_chinese_fonts:
                try:
                    chinese_font = fm.FontProperties(family=font_name)
                    break
                except:
                    continue
    except:
        chinese_font = None
    
    # Only filter font-related warnings when drawing text, keep all warnings for other operations
    fig, ax = plt.subplots(1, figsize=(15, 10))
    ax.imshow(image)
    ax.axis('off')
    
    shown_labels = set()
    text_positions = []
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, item in enumerate(bbox_data):
        if not isinstance(item, dict) or 'label' not in item or 'zoom_in_2d' in item or 'depth' in item:
            continue
        
        color = colors[i % len(colors)]
        
        # Only filter font warnings when calling functions that may use fonts
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*missing from font.*")
            if 'point_2d' in item:
                draw_point_annotation(ax, item, i, color, width, height, text_positions, shown_labels, chinese_font)
            elif 'bbox_2d' in item:
                draw_bbox_annotation(ax, item, i, color, width, height, text_positions, shown_labels, chinese_font)
            elif 'line_2d' in item:
                draw_line_annotation(ax, item, i, color, width, height, text_positions, shown_labels, chinese_font)

    legend_elements = []
    for i, item in enumerate(bbox_data):
        if 'zoom_in_2d' not in item:
            color = colors[i % len(colors)]
            legend_elements.append(plt.Line2D([0], [0], color=color, lw=3, label=f'Evidence {i+1}'))
    
    # Legend drawing may also involve fonts, so filter font warnings as well
    if len(legend_elements) <= 10:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*missing from font.*")
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))

    buf = BytesIO()
    # Saving image may also trigger font warnings, need to filter
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=".*missing from font.*")
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.2, dpi=150)
    plt.close(fig)
    buf.seek(0)
    plotted_image = Image.open(buf).convert("RGB")
    buf.close()
    
    return plotted_image

# ==================== Depth Estimation Functions ====================

# Model cache: avoid repeated loading of MiDaS model (performance optimization)
_depth_model_cache = {}  # Format: {(model_type): (midas_model, device, transform)}

def _load_transforms_offline(hub_cache_dir: str, model_type: str):
    """
    Load transforms from local cache directory, completely offline
    
    Args:
        hub_cache_dir: torch.hub cache directory
        model_type: Model type
    
    Returns:
        Transform object
    """
    import sys
    import os
    
    midas_dir = os.path.join(hub_cache_dir, 'intel-isl_MiDaS_master')
    
    # Check if cached MiDaS code exists locally
    if not os.path.exists(midas_dir):
        raise FileNotFoundError(
            f"MiDaS code not found at {midas_dir}. "
            f"Please run: python download_depth_model.py --preload-transforms"
        )
    
    # Add midas directory to Python path
    if midas_dir not in sys.path:
        sys.path.insert(0, midas_dir)
    
    try:
        # Import transforms directly from local cache
        from torchvision.transforms import Compose
        from midas.transforms import Resize, NormalizeImage, PrepareForNet
        import midas.transforms as transforms_module
        import torch
        import cv2
        
        # Create transforms object (based on hubconf.py logic)
        if model_type in ["DPT_Large", "DPT_Hybrid"]:
            transform = Compose([
                lambda img: {"image": img / 255.0},
                Resize(
                    384, 384,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="minimal",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                PrepareForNet(),
                lambda sample: torch.from_numpy(sample["image"]).unsqueeze(0),
            ])
        else:
            transform = Compose([
                lambda img: {"image": img / 255.0},
                Resize(
                    256, 256,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="upper_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
                lambda sample: torch.from_numpy(sample["image"]).unsqueeze(0),
            ])
        
        return transform
    except ImportError as e:
        raise ImportError(
            f"Failed to load transforms from local cache: {e}. "
            f"Please ensure .torch_hub directory contains complete MiDaS code."
        )

def _load_model_architecture_offline(hub_cache_dir: str, model_type: str):
    """
    Load model architecture from local cache directory, completely offline
    
    Args:
        hub_cache_dir: torch.hub cache directory
        model_type: Model type
    
    Returns:
        Model object (without loaded weights)
    """
    import sys
    import os
    
    midas_dir = os.path.join(hub_cache_dir, 'intel-isl_MiDaS_master')
    
    # Check if cached MiDaS code exists locally
    if not os.path.exists(midas_dir):
        raise FileNotFoundError(
            f"MiDaS code not found at {midas_dir}. "
            f"Please run: python download_depth_model.py --preload-transforms"
        )
    
    # Add midas directory to Python path
    if midas_dir not in sys.path:
        sys.path.insert(0, midas_dir)
    
    try:
        # Import model architecture directly from local cache
        from midas.dpt_depth import DPTDepthModel
        from midas.midas_net import MidasNet
        from midas.midas_net_custom import MidasNet_small
        
        # Create architecture based on model type
        if model_type == "DPT_Large":
            model = DPTDepthModel(
                path=None,
                backbone="vitl16_384",
                non_negative=True,
            )
        elif model_type == "DPT_Hybrid":
            model = DPTDepthModel(
                path=None,
                backbone="vitb_rn50_384",
                non_negative=True,
            )
        elif model_type == "MiDaS":
            model = MidasNet(path=None, non_negative=True)
        elif model_type == "MiDaS_small":
            model = MidasNet_small(path=None, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model
    except ImportError as e:
        raise ImportError(
            f"Failed to load model architecture from local cache: {e}. "
            f"Please ensure .torch_hub directory contains complete MiDaS code."
        )

def _get_cached_depth_model(model_type: str = "DPT_Large", model_path: Optional[str] = None):
    """
    Get cached MiDaS model, load and cache if not exists
    Prefer local cache, work completely offline
    
    Args:
        model_type: Model type ('DPT_Large', 'DPT_Hybrid', 'MiDaS', 'MiDaS_small')
        model_path: Local model file path (optional)
    
    Returns:
        (midas_model, device, transform): Cached MiDaS model, device and transform
    """
    import torch
    import os
    
    # Set torch.hub cache directory to current working directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    hub_cache_dir = os.path.join(current_dir, '.torch_hub')
    os.makedirs(hub_cache_dir, exist_ok=True)
    torch.hub.set_dir(hub_cache_dir)
    
    # Use model path as part of cache key
    cache_key = f"{model_type}_{model_path}" if model_path else model_type
    
    # Check cache
    if cache_key in _depth_model_cache:
        return _depth_model_cache[cache_key]
    
    # If local model path is provided, try loading from local file
    if model_path and os.path.exists(model_path):
        try:
            print(f"Loading depth model from local file: {model_path}")
            # Try loading model (could be torch.jit or torch model)
            midas = None
            try:
                # First try loading as JIT model
                midas = torch.jit.load(model_path, map_location='cpu')
                midas.eval()
                print("  ✓ Loaded as JIT model")
            except Exception as jit_error:
                # If not a jit model, try loading as regular model
                try:
                    checkpoint = torch.load(model_path, map_location='cpu')
                    # Check if it's state_dict format
                    if isinstance(checkpoint, dict) and 'model' in checkpoint:
                        # If it's state_dict, need to load model architecture first
                        print("  - Model file contains state_dict, loading architecture from local cache (offline)...")
                        # Prefer loading architecture from local cache
                        midas = _load_model_architecture_offline(hub_cache_dir, model_type)
                        # Then load weights
                        if 'model_state_dict' in checkpoint:
                            midas.load_state_dict(checkpoint['model_state_dict'])
                        elif 'state_dict' in checkpoint:
                            midas.load_state_dict(checkpoint['state_dict'])
                        else:
                            midas.load_state_dict(checkpoint)
                        midas.eval()
                        print("  ✓ Loaded state_dict into model architecture")
                    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        # Prefer loading architecture from local cache
                        midas = _load_model_architecture_offline(hub_cache_dir, model_type)
                        midas.load_state_dict(checkpoint['state_dict'])
                        midas.eval()
                        print("  ✓ Loaded state_dict")
                    else:
                        # Check if it's state_dict (OrderedDict)
                        if isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
                            # Check if model file contains BEiT (different architecture)
                            model_filename = os.path.basename(model_path).lower()
                            if 'beit' in model_filename:
                                print(f"  ⚠ Warning: {model_filename} appears to be a DPT-BEiT model, which has a different architecture than DPT_Large.")
                                print(f"  ⚠ This model cannot be loaded with the standard DPT_Large architecture.")
                                print(f"  ⚠ Falling back to downloading standard DPT_Large model from hub...")
                                raise ValueError(f"Model architecture mismatch: BEiT model cannot be loaded as DPT_Large")
                            
                            # This might be state_dict, need to load model architecture first
                            print("  - Detected state_dict format, loading architecture from local cache (offline)...")
                            try:
                                # Prefer loading architecture from local cache
                                midas = _load_model_architecture_offline(hub_cache_dir, model_type)
                                print("  ✓ Model architecture loaded from local cache")
                                # Try loading weights, will raise error if keys don't match
                                missing_keys, unexpected_keys = midas.load_state_dict(checkpoint, strict=False)
                                if missing_keys:
                                    print(f"  ⚠ Warning: Missing keys in state_dict: {len(missing_keys)} keys")
                                    if len(missing_keys) > 10:
                                        print(f"  ⚠ First few missing keys: {missing_keys[:5]}")
                                    # If too many keys are missing, architecture mismatch
                                    if len(missing_keys) > 50:
                                        raise ValueError(f"Too many missing keys ({len(missing_keys)}), architecture mismatch")
                                if unexpected_keys:
                                    print(f"  ⚠ Warning: Unexpected keys in state_dict: {len(unexpected_keys)} keys")
                                midas.eval()
                                print("  ✓ Loaded state_dict into model architecture")
                            except Exception as load_error:
                                print(f"  ✗ Failed to load state_dict: {load_error}")
                                raise load_error
                        else:
                            # Directly a model object
                            midas = checkpoint
                            if hasattr(midas, 'eval'):
                                midas.eval()
                            print("  ✓ Loaded as model object")
                except Exception as load_error:
                    print(f"  ✗ JIT load failed: {jit_error}")
                    print(f"  ✗ Regular load failed: {load_error}")
                    raise load_error
            
            if midas is None:
                raise ValueError("Failed to load model from file")
            
            # Load transforms from local cache (completely offline)
            print("  - Loading transforms from local cache (offline)...")
            transform = _load_transforms_offline(hub_cache_dir, model_type)
            print("  ✓ Transforms loaded from local cache")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            midas.to(device=device)
            print(f"  ✓ Model loaded successfully on {device}")
            
            # Cache model
            _depth_model_cache[cache_key] = (midas, device, transform)
            
            return midas, device, transform
        except Exception as e:
            print(f"✗ Failed to load model from local path {model_path}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(
                f"Failed to load model from local path (offline mode). "
                f"Please check if the model file exists and has correct format."
            ) from e
    
    # Try loading MiDaS model from torch hub (prefer local cache)
    try:
        # First try loading from local cache (completely offline)
        midas_dir = os.path.join(hub_cache_dir, 'intel-isl_MiDaS_master')
        if os.path.exists(midas_dir):
            print(f"Loading depth model from local cache (offline, model_type: {model_type})...")
            try:
                midas = _load_model_architecture_offline(hub_cache_dir, model_type)
                # If pretrained weights exist, try loading from checkpoints directory
                checkpoint_dir = os.path.join(hub_cache_dir, 'checkpoints')
                checkpoint_file = None
                if model_type == "DPT_Large":
                    checkpoint_file = os.path.join(checkpoint_dir, 'dpt_large_384.pt')
                elif model_type == "DPT_Hybrid":
                    checkpoint_file = os.path.join(checkpoint_dir, 'dpt_hybrid_384.pt')
                elif model_type == "MiDaS":
                    checkpoint_file = os.path.join(checkpoint_dir, 'midas_v21_384.pt')
                elif model_type == "MiDaS_small":
                    checkpoint_file = os.path.join(checkpoint_dir, 'midas_v21_small_256.pt')
                
                if checkpoint_file and os.path.exists(checkpoint_file):
                    print(f"  - Loading weights from: {checkpoint_file}")
                    checkpoint = torch.load(checkpoint_file, map_location='cpu')
                    if isinstance(checkpoint, dict):
                        if 'state_dict' in checkpoint:
                            midas.load_state_dict(checkpoint['state_dict'])
                        else:
                            midas.load_state_dict(checkpoint)
                    else:
                        midas.load_state_dict(checkpoint)
                    print("  ✓ Weights loaded from local checkpoint")
                else:
                    print("  ⚠ No local checkpoint found, using untrained model")
                
                midas.eval()
                
                # Load transforms from local cache
                transform = _load_transforms_offline(hub_cache_dir, model_type)
                print("  ✓ Transforms loaded from local cache")
            except Exception as offline_error:
                print(f"  ✗ Failed to load from local cache: {offline_error}")
                raise RuntimeError(
                    f"Failed to load model from local cache (offline mode). "
                    f"Please ensure you have run: python download_depth_model.py --preload-transforms"
                ) from offline_error
        else:
            # Local cache does not exist, download not allowed in completely offline mode
            raise FileNotFoundError(
                f"MiDaS code not found at {midas_dir}. "
                f"Completely offline mode requires local cache. "
                f"Please run: python download_depth_model.py --preload-transforms"
            )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        midas.to(device=device)
        
        # Cache model
        _depth_model_cache[cache_key] = (midas, device, transform)
        
        return midas, device, transform
    except Exception as e:
        raise RuntimeError(f"Failed to load MiDaS model: {e}")

def apply_depth_estimation(image: Image.Image,
                          depth_model_path: Optional[str] = None,
                          depth_model_name: Optional[str] = None,
                          config: Optional[Dict] = None) -> Image.Image:
    """
    Apply depth estimation operation to image, call external depth estimation model
    
    Args:
        image: Input PIL Image object
        depth_model_path: Model path (optional, read from config)
        depth_model_name: Model name, default 'midas' (optional)
        config: Configuration dictionary (optional)
    
    Returns:
        PIL Image object containing depth map visualization
    
    Examples:
        # Use default MiDaS model
        depth_map = apply_depth_estimation(image)
        
        # Use configuration
        depth_map = apply_depth_estimation(image, config={'depth_model_name': 'DPT_Large'})
    """
    import os
    
    # Get depth configuration from config (if provided)
    # Support two formats:
    # 1. Flat format: {'depth_model_path': ..., 'depth_model_name': ...}
    # 2. Nested format: {'depth': {'model_path': ..., 'model_name': ...}}
    if config:
        # First try flat format
        temp_path = config.get('depth_model_path', depth_model_path)
        if temp_path:
            # Expand user home directory and resolve relative paths
            temp_path = os.path.expanduser(temp_path)
            if not os.path.isabs(temp_path):
                temp_path = os.path.abspath(temp_path)
            depth_model_path = temp_path
        depth_model_name = config.get('depth_model_name', depth_model_name)
        
        # If flat format doesn't exist, try nested format
        if 'depth' in config and isinstance(config['depth'], dict):
            depth_config = config['depth']
            if not depth_model_path:
                temp_path = depth_config.get('model_path', depth_model_path)
                if temp_path:
                    # Expand user home directory and resolve relative paths
                    temp_path = os.path.expanduser(temp_path)
                    if not os.path.isabs(temp_path):
                        temp_path = os.path.abspath(temp_path)
                    depth_model_path = temp_path
            if not depth_model_name:
                depth_model_name = depth_config.get('model_name', depth_model_name)
    
    # If no model information provided, try using default configuration
    if not depth_model_name:
        depth_model_name = 'DPT_Large'  # Default to DPT_Large
    
    # Check if default paths exist (skip BEiT models due to architecture incompatibility)
    if not depth_model_path:
        default_paths = [
            './models/dpt_large_384.pt',
            './models/dpt_large.pt',
            './models/dpt_hybrid_384.pt',
            './models/midas_v21_384.pt',
            './models/midas_v21_small_256.pt'
        ]
        for path in default_paths:
            expanded_path = os.path.expanduser(path)
            if not os.path.isabs(expanded_path):
                expanded_path = os.path.abspath(expanded_path)
            if os.path.exists(expanded_path):
                # Skip BEiT models (architecture incompatible)
                if 'beit' in os.path.basename(expanded_path).lower():
                    print(f"Skipping BEiT model (architecture incompatible): {expanded_path}")
                    continue
                depth_model_path = expanded_path
                print(f"Found local depth model: {depth_model_path}")
                break
    
    # Call external depth model
    try:
        depth_image = _call_depth_model(
            image,
            model_path=depth_model_path,
            model_name=depth_model_name
        )
        return depth_image
    except Exception as e:
        print(f"Warning: Depth estimation failed: {type(e).__name__}: {e}")
        # If depth fails, use fallback method
        return _depth_fallback(image)

def _call_depth_model(image: Image.Image,
                     model_path: Optional[str] = None,
                     model_name: Optional[str] = None) -> Image.Image:
    """
    Internal function: call actual depth estimation model
    
    Supports multiple depth estimation models:
    - MiDaS (Intel's MiDaS) - High quality depth estimation
    - Fallback method - Simple depth estimation based on edge detection
    
    Args:
        image: Input PIL Image object
        model_path: Model file path (optional, MiDaS loaded from torch hub)
        model_name: Model name ('DPT_Large', 'DPT_Hybrid', 'MiDaS', 'MiDaS_small')
    
    Returns:
        Depth map PIL Image object
    """
    import numpy as np
    
    # Try importing torch-related libraries
    try:
        import torch
        TORCH_AVAILABLE = True
    except ImportError as e:
        TORCH_AVAILABLE = False
        print(f"Warning: torch not available: {e}")
        return _depth_fallback(image)
    
    # If torch is available, try using MiDaS
    if TORCH_AVAILABLE:
        # Model type mapping
        model_type_map = {
            'midas': 'DPT_Large',
            'midas_dpt_large': 'DPT_Large',
            'midas_dpt_hybrid': 'DPT_Hybrid',
            'midas_small': 'MiDaS_small',
            'dpt_large': 'DPT_Large',
            'dpt_hybrid': 'DPT_Hybrid',
            'DPT_Large': 'DPT_Large',
            'DPT_Hybrid': 'DPT_Hybrid',
            'MiDaS': 'MiDaS',
            'MiDaS_small': 'MiDaS_small'
        }
        
        model_type = model_type_map.get(model_name, 'DPT_Large') if model_name else 'DPT_Large'
        
        try:
            return _depth_with_midas(image, model_type, model_path=model_path)
        except Exception as e:
            print(f"Warning: MiDaS depth estimation failed: {type(e).__name__}: {e}")
            return _depth_fallback(image)
    
    # Otherwise use fallback method
    return _depth_fallback(image)

def _depth_with_midas(image: Image.Image, model_type: str = "DPT_Large", model_path: Optional[str] = None) -> Image.Image:
    """
    Use MiDaS (Intel's MiDaS) for depth estimation
    
    Args:
        image: Input PIL Image object
        model_type: Model type ('DPT_Large', 'DPT_Hybrid', 'MiDaS', 'MiDaS_small')
        model_path: Local model file path (optional)
    
    Returns:
        Depth map PIL Image object
    """
    import torch
    import numpy as np
    
    # Get or load MiDaS model from cache
    try:
        midas, device, transform = _get_cached_depth_model(model_type, model_path=model_path)
    except Exception as e:
        print(f"Warning: Failed to load MiDaS model: {type(e).__name__}: {e}")
        raise
    
    # Preprocess image
    img_array = np.array(image)
    if len(img_array.shape) == 2:  # Grayscale image
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Apply transforms
    # The last lambda in transform has already called unsqueeze(0), so no need to unsqueeze again
    try:
        transformed = transform(img_array)
        # If transform returns tuple, take first element
        if isinstance(transformed, (tuple, list)):
            print(f"Warning: transform returned {len(transformed)} values, using first one")
            input_batch = transformed[0].to(device)
        else:
            input_batch = transformed.to(device)
    except Exception as e:
        print(f"Error in transform: {type(e).__name__}: {e}")
        raise
    
    # Inference
    with torch.no_grad():
        prediction = midas(input_batch)
        # If model returns tuple, take first element
        if isinstance(prediction, (tuple, list)):
            prediction = prediction[0]
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_array.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    # Convert to numpy and normalize
    depth_map = prediction.cpu().numpy()
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    depth_map = (depth_map * 255).astype(np.uint8)
    
    # Apply color mapping to make depth map more intuitive
    depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
    depth_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
    depth_image = Image.fromarray(depth_rgb)
    
    # Add title
    draw = ImageDraw.Draw(depth_image)
    try:
        font = ImageFont.truetype("Arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    title = f"Depth Map (MiDaS {model_type})"
    try:
        text_bbox = draw.textbbox((0, 0), title, font=font)
        text_width = text_bbox[2] - text_bbox[0]
    except:
        text_width = len(title) * 12
    
    title_height = 30
    title_bg = Image.new('RGB', (depth_image.width, title_height), 'white')
    title_draw = ImageDraw.Draw(title_bg)
    title_x = (depth_image.width - text_width) // 2
    title_draw.text((title_x, 5), title, fill='black', font=font)
    
    final_image = Image.new('RGB', (depth_image.width, depth_image.height + title_height))
    final_image.paste(title_bg, (0, 0))
    final_image.paste(depth_image, (0, title_height))
    
    return final_image

def _depth_fallback(image: Image.Image) -> Image.Image:
    """
    Fallback depth estimation method: use simple depth estimation based on edge detection when MiDaS is unavailable
    """
    import numpy as np
    
    # 转换为numpy数组
    img_array = np.array(image)
    
    # 转换为灰度图
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # 基于边缘检测的深度估计
    edges = cv2.Canny(gray, 50, 150)
    dist_transform = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
    depth = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply color mapping to make depth map more intuitive
    depth_colored = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
    depth_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
    depth_image = Image.fromarray(depth_rgb)
    
    # Add title
    draw = ImageDraw.Draw(depth_image)
    try:
        font = ImageFont.truetype("Arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    title = "Depth Map (Fallback)"
    try:
        text_bbox = draw.textbbox((0, 0), title, font=font)
        text_width = text_bbox[2] - text_bbox[0]
    except:
        text_width = len(title) * 12
    
    title_height = 30
    title_bg = Image.new('RGB', (depth_image.width, title_height), 'white')
    title_draw = ImageDraw.Draw(title_bg)
    title_x = (depth_image.width - text_width) // 2
    title_draw.text((title_x, 5), title, fill='black', font=font)
    
    final_image = Image.new('RGB', (depth_image.width, depth_image.height + title_height))
    final_image.paste(title_bg, (0, 0))
    final_image.paste(depth_image, (0, title_height))
    
    return final_image

def process_depth_operations(image: Image.Image, bbox_data: List[Dict], config: Optional[Dict] = None) -> List[Image.Image]:
    """
    Process depth operations, extract depth requests from bbox_data and apply them
    Note: Only process the first depth request, ignore subsequent depth requests
    
    Args:
        image: Input PIL Image object
        bbox_data: List of data containing depth field
        config: Optional configuration dictionary containing depth model configuration
    
    Returns:
        List of depth maps (at most 1 image)
    """
    for item in bbox_data:
        if not isinstance(item, dict):
            continue
        
        # Check if depth field exists (only process the first one)
        if 'depth' in item:
            try:
                depth_image = apply_depth_estimation(
                    image,
                    config=config
                )
                return [depth_image]
            except Exception as e:
                print(f"Warning: Depth operation failed: {e}")
                # Return even on failure, do not process subsequent depth requests
                return []
    
    return []

# ==================== Segmentation Functions ====================

# Model cache: avoid repeated loading of SAM model (performance optimization)
_segmentation_model_cache = {}  # Format: {(model_path, model_type): (sam_model, device)}

def _get_cached_sam_model(model_path: str, model_type: str):
    """
    Get cached SAM model, load and cache if not exists
    
    Args:
        model_path: Model file path
        model_type: Model type ('vit_b', 'vit_l', 'vit_h')
    
    Returns:
        (sam_model, device): Cached SAM model and device
    """
    import torch
    import os
    from segment_anything import sam_model_registry
    
    # Normalize path (ensure relative and absolute paths pointing to the same file use the same cache)
    normalized_path = os.path.abspath(os.path.expanduser(model_path))
    cache_key = (normalized_path, model_type)
    
    # Check cache
    if cache_key in _segmentation_model_cache:
        return _segmentation_model_cache[cache_key]
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=normalized_path)
    sam.to(device=device)
    
    # Cache model
    _segmentation_model_cache[cache_key] = (sam, device)
    
    return sam, device

def apply_segmentation(image: Image.Image, 
                      bbox: Optional[List[float]] = None,
                      point: Optional[Union[List[float], List[List[float]]]] = None,
                      points: Optional[List[List[float]]] = None,
                      point_labels: Optional[List[int]] = None,
                      segmentation_model_path: Optional[str] = None,
                      segmentation_model_name: Optional[str] = None,
                      config: Optional[Dict] = None) -> Image.Image:
    """
    Apply segmentation operation to image, call external segmentation model
    
    Args:
        image: Input PIL Image object
        bbox: Normalized bounding box [x, y, w, h] (optional)
        point: Normalized point coordinates [x, y] (optional)
        points: Multiple normalized point coordinates [[x, y], ...] (optional)
        segmentation_model_path: Model path (optional, read from config)
        segmentation_model_name: Model name, default 'sam' (optional)
        config: Configuration dictionary (optional)
    
    Returns:
        PIL Image object containing segmentation mask overlay
    
    Examples:
        # Use point
        segmented = apply_segmentation(image, point=[0.5, 0.5])
        
        # Use bounding box
        segmented = apply_segmentation(image, bbox=[0.2, 0.2, 0.4, 0.4])
        
        # Use multiple points
        segmented = apply_segmentation(image, points=[[0.3, 0.3], [0.7, 0.7]])
    """
    import os
    
    # Get segmentation configuration from config (if provided)
    if config:
        segmentation_model_path = config.get('segmentation_model_path', segmentation_model_path)
        segmentation_model_name = config.get('segmentation_model_name', segmentation_model_name)
    
    # If model information is not provided, try to use default configuration
    if not segmentation_model_path and not segmentation_model_name:
        # Try to get from environment variables or default paths
        segmentation_model_name = 'sam'  # Default to SAM
        segmentation_model_path = './models/sam_vit_b.pth'
    
    # Convert normalized coordinates to pixel coordinates
    img_width, img_height = image.size
    prompt_box = None
    prompt_boxes = None  # Support multiple bboxes
    prompt_point = None
    prompt_point_labels = None
    
    if bbox:
        # Check if it's a list of multiple bboxes
        if isinstance(bbox, (list, tuple)) and len(bbox) > 0:
            # Check if first element is a list (multiple bboxes case)
            if isinstance(bbox[0], (list, tuple)):
                # Multiple bboxes: [[x, y, w, h], ...]
                prompt_boxes = []
                for b in bbox:
                    if isinstance(b, (list, tuple)) and len(b) == 4:
                        x_norm, y_norm, w_norm, h_norm = b
                        x1 = int(float(x_norm) * img_width)
                        y1 = int(float(y_norm) * img_height)
                        x2 = int((float(x_norm) + float(w_norm)) * img_width)
                        y2 = int((float(y_norm) + float(h_norm)) * img_height)
                        prompt_boxes.append([x1, y1, x2, y2])
                # If only one bbox, convert to single bbox format for compatibility
                if len(prompt_boxes) == 1:
                    prompt_box = prompt_boxes[0]
                    prompt_boxes = None
            elif len(bbox) == 4:
                # Single bbox: [x, y, w, h]
                x_norm, y_norm, w_norm, h_norm = bbox
                x1 = int(float(x_norm) * img_width)
                y1 = int(float(y_norm) * img_height)
                x2 = int((float(x_norm) + float(w_norm)) * img_width)
                y2 = int((float(y_norm) + float(h_norm)) * img_height)
                prompt_box = [x1, y1, x2, y2]
    
    # Process point prompts: support single point or multiple points
    if points is not None:
        # Use points parameter (multiple points)
        prompt_point = []
        for pt in points:
            if len(pt) == 2:
                x_norm, y_norm = pt
                x = int(x_norm * img_width)
                y = int(y_norm * img_height)
                prompt_point.append([x, y])
        # Use provided point_labels, default to foreground points (1) if not provided
        if point_labels is not None and len(point_labels) == len(prompt_point):
            prompt_point_labels = point_labels
        else:
            prompt_point_labels = [1] * len(prompt_point)  # Default all to foreground points
    elif point is not None:
        # Use point parameter (single point or multiple points)
        if isinstance(point[0], (list, tuple)):
            # point is a list of points [[x, y], ...]
            prompt_point = []
            for pt in point:
                if len(pt) == 2:
                    x_norm, y_norm = pt
                    x = int(x_norm * img_width)
                    y = int(y_norm * img_height)
                    prompt_point.append([x, y])
            # If point_labels not provided, default to foreground points
            if point_labels is not None and len(point_labels) == len(prompt_point):
                prompt_point_labels = point_labels
            else:
                prompt_point_labels = [1] * len(prompt_point)
        else:
            # point is a single point [x, y]
            if len(point) == 2:
                x_norm, y_norm = point
                x = int(x_norm * img_width)
                y = int(y_norm * img_height)
                prompt_point = [[x, y]]
                prompt_point_labels = [1]  # Default to foreground point
    
    # Call external segmentation model
    try:
        # If there are multiple bboxes, process each bbox separately and merge results
        if prompt_boxes and len(prompt_boxes) > 1:
            # Segment each bbox separately, then merge all masks
            segmented_image = _call_segmentation_model_multiple_boxes(
                image,
                prompt_boxes=prompt_boxes,
                prompt_point=prompt_point,
                prompt_point_labels=prompt_point_labels,
                model_path=segmentation_model_path,
                model_name=segmentation_model_name
            )
        else:
            # Single bbox or no bbox, use original logic
            segmented_image = _call_segmentation_model(
                image, 
                prompt_box=prompt_box,
                prompt_point=prompt_point,
                prompt_point_labels=prompt_point_labels,
                model_path=segmentation_model_path,
                model_name=segmentation_model_name
            )
        return segmented_image
    except Exception as e:
        print(f"Warning: Segmentation failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        # If segmentation fails, return original image
        return None

def _call_segmentation_model(image: Image.Image,
                            prompt_box: Optional[List[int]] = None,
                            prompt_point: Optional[List[List[int]]] = None,
                            prompt_point_labels: Optional[List[int]] = None,
                            model_path: Optional[str] = None,
                            model_name: Optional[str] = None) -> Image.Image:
    """
    Internal function: call actual segmentation model
    
    Supports multiple segmentation models:
    - SAM (Segment Anything Model) - supports multiple point prompts (foreground and background points)
    - Other custom models
    
    Args:
        image: Input PIL Image object
        prompt_box: Bounding box prompt [x1, y1, x2, y2] (pixel coordinates)
        prompt_point: Point prompt list [[x, y], ...] (pixel coordinates)
        prompt_point_labels: Point label list [0, 1, ...], 0 for background points, 1 for foreground points
        model_path: Model file path
        model_name: Model name
    
    Returns:
        Image with segmentation mask overlay
    """
    import numpy as np
    
    # Try to import SAM-related libraries
    try:
        from segment_anything import sam_model_registry, SamPredictor
        SAM_AVAILABLE = True
    except ImportError as e:
        SAM_AVAILABLE = False
        print(f"Warning: segment_anything not available: {e}")
        return _segment_fallback(image, prompt_box, prompt_point)
    
    # If SAM is available, use SAM
    if SAM_AVAILABLE and (model_name and 'sam' in model_name.lower()):
        try:
            return _segment_with_sam(image, prompt_box, prompt_point, prompt_point_labels, model_path, model_name)
        except Exception as e:
            print(f"Warning: SAM segmentation failed: {type(e).__name__}: {e}")
            return _segment_fallback(image, prompt_box, prompt_point)
    
    # Otherwise use simple fallback method (simple bbox-based mask)
    return _segment_fallback(image, prompt_box, prompt_point)

def _segment_with_sam(image: Image.Image,
                     prompt_box: Optional[List[int]] = None,
                     prompt_point: Optional[List[List[int]]] = None,
                     prompt_point_labels: Optional[List[int]] = None,
                     model_path: Optional[str] = None,
                     model_name: Optional[str] = None) -> Image.Image:
    """
    Use SAM (Segment Anything Model) for segmentation
    
    Supports multiple prompt methods:
    - Single or multiple points (foreground and background points)
    - Bounding box
    - Point + bounding box combination
    """
    from segment_anything import sam_model_registry, SamPredictor
    import numpy as np
    import torch
    
    import os
    
    # If path is not provided, try to use default paths
    if model_path is None:
        # Try common SAM model paths
        possible_paths = [
            f'./models/vit_h.pth',
            f'./checkpoints/vit_h.pth',
            f'~/.cache/sam/vit_h.pth'
        ]
        for path in possible_paths:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                model_path = expanded_path
                break
        
        if model_path is None:
            raise FileNotFoundError(f"SAM model not found. Please specify model_path in config.")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"SAM model file not found: {model_path}")
    
    # Automatically detect model type from filename (if model_name is not specified or is generic)
    # Prefer information from filename as it is more accurate
    filename = os.path.basename(model_path).lower()
    auto_detected_type = None
    if 'vit_b' in filename or 'sam_vit_b' in filename:
        auto_detected_type = 'vit_b'
    elif 'vit_l' in filename or 'sam_vit_l' in filename:
        auto_detected_type = 'vit_l'
    elif 'vit_h' in filename or 'sam_vit_h' in filename:
        auto_detected_type = 'vit_h'
    
    # Determine model type and path
    if model_name is None:
        if auto_detected_type:
            model_name = f'sam_{auto_detected_type}'
        else:
            model_name = 'sam_vit_h'
    
    # Model type mapping - SAM registry uses keys 'vit_b', 'vit_l', 'vit_h'
    model_type_map = {
        'sam': 'vit_h',
        'sam_vit_h': 'vit_h',
        'sam_vit_l': 'vit_l',
        'sam_vit_b': 'vit_b',
        'vit_h': 'vit_h',
        'vit_l': 'vit_l',
        'vit_b': 'vit_b'
    }
    
    model_type = model_type_map.get(model_name.lower(), 'vit_h')
    
    # If detected type from filename doesn't match specified model_name, prefer filename detection result
    if auto_detected_type and model_type != auto_detected_type:
        print(f"Warning: Model name '{model_name}' suggests type '{model_type}', but filename indicates '{auto_detected_type}'. Using '{auto_detected_type}' from filename.")
        model_type = auto_detected_type
    
    # Verify model type is in registry
    if model_type not in sam_model_registry:
        available_models = list(sam_model_registry.keys())
        raise KeyError(f"Model type '{model_type}' not found in sam_model_registry. Available models: {available_models}")
    
    # Get or load SAM model from cache (performance optimization: avoid repeated loading)
    try:
        sam, device = _get_cached_sam_model(model_path, model_type)
        predictor = SamPredictor(sam)
    except Exception as e:
        print(f"Warning: Failed to load SAM model: {type(e).__name__}: {e}")
        raise
    
    # Convert image to numpy array
    image_array = np.array(image)
    
    # Set image
    predictor.set_image(image_array)
    
    # Prepare prompts
    point_coords = None
    point_labels_array = None
    input_box = None
    
    # Process point prompts
    if prompt_point:
        point_coords = np.array(prompt_point)
        # Use provided point_labels, default to foreground points (1) if not provided
        if prompt_point_labels is not None and len(prompt_point_labels) == len(prompt_point):
            point_labels_array = np.array(prompt_point_labels)
        else:
            point_labels_array = np.array([1] * len(prompt_point))  # Default all to foreground points
    
    # Process bounding box prompts
    if prompt_box:
        input_box = np.array(prompt_box)[None, :]  # SAM requires shape (1, 4)
    
    # Combine point prompts and bounding box prompts (if both are available)
    if point_coords is not None and input_box is not None:
        # Use both points and bounding box
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels_array,
            box=input_box,
            multimask_output=False,
        )
    elif input_box is not None:
        # Use only bounding box
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box,
            multimask_output=False,
        )
    elif point_coords is not None:
        # Use only point prompts (supports multiple points, including foreground and background points)
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels_array,
            box=None,
            multimask_output=False,
        )
    else:
        # If no prompts, use entire image
        h, w = image_array.shape[:2]
        input_box = np.array([0, 0, w, h])[None, :]
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box,
            multimask_output=False,
        )
    
    # Get best mask
    mask = masks[0]
    
    # Ensure mask is uint8 format (0-255)
    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255
    elif mask.dtype != np.uint8:
        # If mask is float (0-1), convert to uint8 (0-255)
        if mask.max() <= 1.0:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)
    
    # Overlay mask on original image
    result_image = _overlay_mask_on_image(image, mask)
    
    return result_image

def _segment_fallback(image: Image.Image,
                     prompt_box: Optional[List[int]] = None,
                     prompt_point: Optional[List[List[int]]] = None) -> Image.Image:
    """
    Fallback segmentation method: use simple bbox-based mask when SAM is unavailable
    """
    import numpy as np
    
    # Create mask
    mask = np.zeros((image.height, image.width), dtype=np.uint8)
    
    if prompt_box:
        # Use bbox to create rectangular mask
        x1, y1, x2, y2 = prompt_box
        mask[y1:y2, x1:x2] = 255
    elif prompt_point:
        # Use point to create circular mask
        for point in prompt_point:
            x, y = point
            # Create a small circular region
            radius = min(image.width, image.height) // 20
            y_coords, x_coords = np.ogrid[:image.height, :image.width]
            mask_circle = (x_coords - x) ** 2 + (y_coords - y) ** 2 <= radius ** 2
            mask[mask_circle] = 255
    else:
        # If no prompts, use entire image
        mask.fill(255)
    
    # Overlay mask on image
    result_image = _overlay_mask_on_image(image, mask)
    
    return result_image

def _overlay_mask_on_image(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """
    Overlay segmentation mask on image, use highlighting and contour lines to make effect more obvious
    
    Args:
        image: Original PIL Image object
        mask: numpy array mask (0-255 or bool)
    
    Returns:
        Visualization image with mask overlay
    """
    import numpy as np
    from PIL import ImageDraw, ImageFont
    
    # Convert image to numpy array
    img_array = np.array(image.convert('RGB'))
    
    # Ensure mask is in correct format and range
    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255
    elif mask.max() <= 1.0:
        # If mask is float in 0-1 range, convert to 0-255
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)
    
    # Normalize mask to 0-1
    mask_normalized = mask.astype(np.float32) / 255.0
    mask_binary = (mask > 128).astype(np.uint8)  # Binarize mask for contour detection
    
    # Check if mask has valid values
    if mask.max() == 0:
        print("Warning: Mask is empty (all zeros), no segmentation to display")
        return image.copy()
    
    # If mask area is too small, also give warning
    mask_area_ratio = np.count_nonzero(mask_binary) / mask_binary.size
    if mask_area_ratio < 0.01:
        print(f"Warning: Mask area is very small: {mask_area_ratio*100:.2f}% of image")
    
    # === SAM Web UI style visualization: only segmented region is bright, other regions are darkened ===
    result = img_array.copy().astype(np.float32)
    mask_3d = mask_normalized[:, :, np.newaxis]
    
    # Method: darken non-mask regions, keep mask regions bright
    # 1. Non-mask regions: reduce brightness (darken)
    darken_factor = 0.3  # Non-mask regions retain 30% brightness (darkened by 70%)
    # 2. Mask regions: maintain or slightly enhance brightness
    brighten_factor = 1.1  # Mask region brightness increased by 10%
    
    # Create result image
    # Non-mask regions: original * darken_factor
    # Mask regions: original * brighten_factor
    result = result * (1 - mask_3d) * darken_factor + result * mask_3d * brighten_factor
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # === Add contour lines (optional, use white or bright colors, more visible on dark background) ===
    # Detect mask boundaries
    try:
        # Try to use scipy (if available, better results)
        from scipy import ndimage
        # Dilate mask
        dilated = ndimage.binary_dilation(mask_binary, structure=np.ones((3, 3)))
        # Boundary = dilated mask - original mask
        boundary = dilated.astype(int) - mask_binary.astype(int)
        boundary = (boundary > 0).astype(np.uint8)
    except ImportError:
        # If scipy is not available, use numpy to implement simple boundary detection
        # Use convolution to detect boundaries
        kernel = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]], dtype=np.float32)
        
        # Manually implement 2D convolution
        h, w = mask_binary.shape
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2
        
        # Padding
        padded = np.pad(mask_binary.astype(np.float32), ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        
        # Convolution
        boundary = np.zeros((h, w), dtype=np.float32)
        for i in range(h):
            for j in range(w):
                boundary[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
        
        # Find boundaries (pixels at mask edges)
        boundary = np.abs(boundary)
        boundary = (boundary > 2).astype(np.uint8) & (mask_binary == 0).astype(np.uint8)
    
    # Draw contour lines (use white or bright yellow, more visible on dark background)
    contour_color = np.array([255, 255, 200])  # Bright yellow/white contour
    contour_width = 2  # Contour line width
    
    # Draw contours
    for c in range(3):
        result[:, :, c] = np.where(boundary > 0, contour_color[c], result[:, :, c])
    
    # If contour is too thin, can slightly dilate
    if contour_width > 1:
        try:
            from scipy import ndimage
            for _ in range(contour_width - 1):
                boundary = ndimage.binary_dilation(boundary, structure=np.ones((3, 3)))
                for c in range(3):
                    result[:, :, c] = np.where(boundary > 0, contour_color[c], result[:, :, c])
        except ImportError:
            # Simple dilation: use max pooling
            kernel_size = 3
            pad = kernel_size // 2
            padded_boundary = np.pad(boundary, pad, mode='constant')
            dilated = np.zeros_like(boundary)
            for i in range(boundary.shape[0]):
                for j in range(boundary.shape[1]):
                    dilated[i, j] = np.max(padded_boundary[i:i+kernel_size, j:j+kernel_size])
            boundary = dilated
            for c in range(3):
                result[:, :, c] = np.where(boundary > 0, contour_color[c], result[:, :, c])
    
    # Convert back to PIL Image
    result_image = Image.fromarray(result)
    
    return result_image

def process_segmentation_operations(image: Image.Image, 
                                   bbox_data: List[Dict],
                                   config: Optional[Dict] = None) -> List[Image.Image]:
    """
    Process segmentation operations, extract segmentation requests from bbox_data and apply them
    Note: Only process the first segmentation request, ignore subsequent segmentation requests
    
    Args:
        image: Input PIL Image object
        bbox_data: List of data containing segmentation_2d field
        config: Optional configuration dictionary containing segmentation model configuration
    
    Returns:
        List of segmentation result images (at most 1 image)
    """
    segmentation_images = []
    
    for item in bbox_data:
        if not isinstance(item, dict):
            continue
        
        # Check if there is segmentation_2d field (only process the first one)
        if 'segmentation_2d' in item:
            seg_data = item['segmentation_2d']
            label = item.get('label', 'Segmentation')
            
            # Extract segmentation parameters - automatically recognize format
            bbox = None
            point = None
            points = None
            
            if isinstance(seg_data, list):
                # List format: automatically recognize
                if len(seg_data) == 4:
                    # [x, y, w, h] - bounding box
                    bbox = seg_data
                elif len(seg_data) == 2:
                    # [x, y] - single point
                    if isinstance(seg_data[0], (int, float)):
                        point = seg_data
                    else:
                        # May be nested format, treat as multiple points
                        points = seg_data
                elif len(seg_data) > 0:
                    # [[x, y], [x, y], ...] - multiple points
                    if isinstance(seg_data[0], list) and len(seg_data[0]) == 2:
                        points = seg_data
                    else:
                        # Other formats, try as single point
                        point = seg_data if len(seg_data) == 2 else None
            elif isinstance(seg_data, dict):
                # Dictionary format: backward compatible, but list format is recommended
                bbox = seg_data.get('bbox')
                point = seg_data.get('point')
                points = seg_data.get('points')
            
            # Apply segmentation (automatic processing, no need to specify point_labels)
            try:
                segmented_image = apply_segmentation(
                    image,
                    bbox=bbox,
                    point=point,
                    points=points,
                    config=config
                )
                segmentation_images.append(segmented_image)
                # Only process the first segmentation request, return immediately after finding it
                return segmentation_images
            except Exception as e:
                print(f"Warning: Segmentation operation failed: {e}")
                # Return even if failed, no longer process subsequent segmentation requests
                return segmentation_images
    
    return segmentation_images

def plot_bounding_boxes(image, bbox_data):
    """
    Main function to process all visual operations
    
    This function processes all visual operations in bbox_data and returns a list of images:
    - Annotated image (contains all bbox/point/line annotations)
    - Zoom image (if zoom_in_2d requested, only process the first one)
    - Segmentation image (if segmentation_2d requested, only process the first one)
    - Depth map (if depth requested, only process the first one)
    
    Args:
        image: PIL Image object
        bbox_data: List of dicts with annotation data
    
    Returns:
        List of PIL Images:
            - Base case: [annotated_image] (all bbox/point/line annotations on one image)
            - With zoom: [annotated_image, zoom_image] (only first zoom)
            - With segmentation: [annotated_image, segmented_image] (only first segmentation)
            - With depth: [annotated_image, depth_map] (only first depth)
            - Combinations are possible (e.g., [annotated_image, zoom_image, segmented_image, depth_map])
    """
    if (bbox_data is None or len(bbox_data) == 0 or image is None):
        return [image] if image else []
        
    result_images = []
    
    # 1. Check if there are bbox/point/line annotations, if so create annotated image
    has_annotation = False
    for item in bbox_data:
        if isinstance(item, dict):
            if 'bbox_2d' in item or 'point_2d' in item or 'line_2d' in item:
                has_annotation = True
                break
    
    if has_annotation:
        annotated_image = create_annotated_image(image, bbox_data)
        result_images.append(annotated_image)
    
    # 2. Process zoom operations (only process the first one)
    zoom_image = process_zoom_operations(image, bbox_data)
    if zoom_image:
        result_images.append(zoom_image)
    
    # 3. Process segmentation operations (only process the first one)
    segmentation_images = process_segmentation_operations(image, bbox_data)
    if segmentation_images:
        result_images.extend(segmentation_images)
    
    # 4. Process depth operations (only process the first one)
    depth_images = process_depth_operations(image, bbox_data)
    if depth_images:
        result_images.extend(depth_images)
    
    return result_images

def combine_images(images: List[Image.Image], layout: str = "horizontal", spacing: int = 10, background_color: str = "white") -> Image.Image:
    """
    Combine multiple PIL images into a single image.
    
    Args:
        images: List of PIL Image objects
        layout: "horizontal" or "vertical" arrangement
        spacing: Pixels of spacing between images
        background_color: Background color for spacing
    
    Returns:
        Combined PIL Image
    """
    if not images:
        raise ValueError("No images provided")
    
    if len(images) == 1:
        return images[0].copy()
    
    rgb_images = []
    for img in images:
        if img.mode != 'RGB':
            rgb_images.append(img.convert('RGB'))
        else:
            rgb_images.append(img.copy())
    
    if layout == "horizontal":
        total_width = sum(img.width for img in rgb_images) + spacing * (len(rgb_images) - 1)
        max_height = max(img.height for img in rgb_images)
        
        combined = Image.new('RGB', (total_width, max_height), background_color)
        
        x_offset = 0
        for img in rgb_images:
            y_offset = (max_height - img.height) // 2
            combined.paste(img, (x_offset, y_offset))
            x_offset += img.width + spacing
            
    elif layout == "vertical":
        max_width = max(img.width for img in rgb_images)
        total_height = sum(img.height for img in rgb_images) + spacing * (len(rgb_images) - 1)
        
        combined = Image.new('RGB', (max_width, total_height), background_color)
        
        y_offset = 0
        for img in rgb_images:
            x_offset = (max_width - img.width) // 2
            combined.paste(img, (x_offset, y_offset))
            y_offset += img.height + spacing
    else:
        raise ValueError(f"Invalid layout '{layout}'. Must be 'horizontal' or 'vertical'")
    
    return combined

def compress_image(image: Union[str, Image.Image], max_size: Tuple[int, int] = (512,512), quality: int = 80) -> Image.Image:
    """
    Compress image to reduce file size
    
    Args:
        image: PIL Image object or path to image file
        max_size: Maximum size (width, height)
        quality: JPEG quality (1-100)
    
    Returns:
        Compressed PIL Image
    """
    if isinstance(image, str):
        image = Image.open(image)
    
    # Convert to RGB if necessary
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')
    
    # Resize if larger than max_size
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    # Compress by saving to BytesIO with specified quality
    buffer = BytesIO()
    image.save(buffer, format='JPEG', quality=quality, optimize=True)
    buffer.seek(0)
    
    return Image.open(buffer)

def encode_image_to_base64(image: Union[str, Image.Image], max_size: Tuple[int, int] = (512,512), quality: int = 80, auto_compress: bool = False) -> str:
    """
    Encode image to base64 string
    
    Args:
        image: PIL Image object or path to image file
        max_size: Maximum size for compression
        quality: JPEG quality for compression
        auto_compress: Whether to auto-compress large images
    
    Returns:
        Base64 encoded string
    """
    if isinstance(image, str):
        image = Image.open(image)
    
    # Auto-compress if requested and image is large
    if auto_compress and (image.width > max_size[0] or image.height > max_size[1]):
        image = compress_image(image, max_size, quality)
    
    # Handle transparency
    if _has_transparency(image):
        # Keep as PNG for transparency
        buffer = BytesIO()
        image.save(buffer, format='PNG')
    else:
        # Convert to RGB and save as JPEG
        if image.mode != 'RGB':
            image = image.convert('RGB')
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
    
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def _has_transparency(image: Image.Image) -> bool:
    """Check if image has transparency"""
    return (
        image.mode in ('RGBA', 'LA') or
        (image.mode == 'P' and 'transparency' in image.info)
    )

