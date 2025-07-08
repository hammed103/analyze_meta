# 🖼️ EV Ad Images Download Guide

## 📊 **Dataset Overview**

Your dataset contains **792 images** from the `new_image_url` column:
- **Total rows**: 7,186
- **Rows with images**: 792 
- **Image format**: 600x600 pixels (Facebook CDN)
- **Image type**: JPG format

## 🚀 **Download Options**

### **1. Quick Test (Already Done ✅)**
```bash
python3 quick_image_download.py
```
- Downloaded **20 sample images** successfully
- Saved to `sample_images/` folder
- **100% success rate** - all URLs are accessible!

### **2. Download All Images (Recommended)**
```bash
python3 download_all_ev_images.py
```
**Features:**
- Downloads all **792 images**
- **Organized by car model** and advertiser
- Creates **thumbnails** (200x200)
- Saves **metadata** for each image
- **Progress tracking** and error handling
- **Respectful delays** between downloads

**Directory Structure Created:**
```
ev_ad_images/
├── by_car_model/          # Images organized by car model
│   ├── Volkswagen_ID_4/
│   ├── Tesla_Model_Y/
│   ├── Hyundai_IONIQ_5/
│   └── ...
├── by_advertiser/         # Images organized by advertiser
├── thumbnails/            # 200x200 thumbnails
├── metadata/              # JSON metadata for each image
├── failed/                # Failed download logs
└── download_statistics.json
```

### **3. Simple Batch Download**
```bash
python3 download_images.py
```
**Features:**
- Interactive options (50, 200, or all images)
- Creates thumbnails automatically
- Detailed error logging
- Progress tracking

## 📁 **What You Get**

### **Image Files:**
- **Original images**: 600x600 pixels, JPG format
- **Thumbnails**: 200x200 pixels for quick viewing
- **Organized folders**: By car model and advertiser

### **Metadata for Each Image:**
```json
{
  "ad_id": "329729626383284",
  "image_url": "https://scontent-ord5-2.xx...",
  "car_model": "Cupra Born",
  "advertiser_name": "SEAT Βελμάρ",
  "page_name": "SEAT Βελμάρ",
  "ad_title": "Ετοιμοπαράδοτο Born",
  "start_date": "2024-04-04 08:00:00",
  "end_date": "2024-09-18 08:00:00",
  "image_size": [600, 600],
  "file_size": 45678,
  "download_date": "2025-07-04T13:30:00"
}
```

### **Statistics Report:**
- Total download success rate
- Images per car model
- Images per advertiser
- Failed download details

## 🎯 **Recommended Workflow**

### **Step 1: Download All Images**
```bash
python3 download_all_ev_images.py
```
*Estimated time: ~10-15 minutes for 792 images*

### **Step 2: Explore the Results**
```bash
# Check the statistics
cat ev_ad_images/download_statistics.json

# See car models with most images
ls -la ev_ad_images/by_car_model/

# Browse thumbnails for quick viewing
open ev_ad_images/thumbnails/
```

### **Step 3: Use in Analysis**
- **Dashboard integration**: Display images in Streamlit
- **Theme analysis**: Visual analysis of ad designs
- **Car model comparison**: Compare visual styles by brand
- **Advertiser analysis**: Study different marketing approaches

## 🔧 **Technical Details**

### **Download Features:**
- ✅ **Respectful delays** (0.5s between downloads)
- ✅ **Error handling** for failed downloads
- ✅ **Progress tracking** every 50 images
- ✅ **Image verification** (checks if valid image)
- ✅ **Duplicate handling** (unique filenames)
- ✅ **Metadata preservation** (ad details saved)

### **File Organization:**
- **Unique filenames**: `{ad_id}_{url_hash}.jpg`
- **Safe directory names**: Special characters removed
- **Metadata linking**: JSON files match image names
- **Thumbnail generation**: Automatic 200x200 thumbnails

### **Error Handling:**
- **Failed downloads logged** to CSV
- **Invalid images removed** automatically
- **Network timeouts handled** gracefully
- **Progress preserved** (can resume if interrupted)

## 📊 **Expected Results**

Based on the successful test download:
- **Success rate**: ~100% (all test images downloaded)
- **Image quality**: 600x600 pixels, good quality
- **Download speed**: ~1-2 images per second
- **Total time**: 10-15 minutes for all 792 images
- **Storage needed**: ~50-100 MB for all images

## 🎨 **Integration with Dashboard**

After downloading, you can:
1. **Add image gallery** to the dashboard
2. **Display images by car model** in theme analysis
3. **Show thumbnails** in search results
4. **Create visual comparisons** between brands
5. **Analyze image themes** with the downloaded images

## 🚀 **Ready to Download?**

Run this command to download all EV ad images:
```bash
python3 download_all_ev_images.py
```

The script will create an organized collection of all 792 EV advertisement images with metadata and thumbnails! 🖼️⚡
