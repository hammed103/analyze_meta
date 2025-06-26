# 🚗⚡ Meta Analysis - Electric Vehicle Ads Dashboard

A comprehensive analysis dashboard for Facebook electric vehicle advertisements, featuring AI-powered image classification and interactive data visualization.

## 📊 Project Overview

This project analyzes Facebook Ads Library data specifically focused on electric vehicle advertisements. It provides insights into advertising strategies, gender targeting, creative themes, and visual content analysis across different EV brands and models.

## 🎯 Key Features

### 📈 Interactive Dashboard
- **Multi-tab Analysis**: Overview, car model deep-dive, creative analysis, image gallery, and advanced analytics
- **Real-time Filtering**: Filter by car models and advertiser types
- **Visual Analytics**: Interactive charts and graphs using Plotly

### 🖼️ Image Analysis
- **AI-Powered Classification**: Automated theme detection and content analysis
- **Visual Gallery**: Paginated image browser with metadata
- **Creative Insights**: Analysis of visual themes and advertising strategies

### 🎯 Targeting Analysis
- **Gender Breakdown**: Male vs female targeting patterns
- **Geographic Targeting**: Country-level advertising distribution
- **Advertiser Classification**: Official brands vs dealers vs third-party

### 🚗 Car Model Focus
Supports analysis of major electric vehicle models:
- Cupra Born
- Mini Aceman E
- Volkswagen ID3, ID.4, ID5
- Kia EV3
- Volvo EX30, EC40
- Renault Megane E-Tech
- MG4
- BMW iX1, iX2, iX3
- Hyundai Ioniq 5
- Audi Q4 e-tron
- Tesla Model Y

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9+
- pip package manager

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Meta\ Analysis
   ```

2. **Install dependencies**
   ```bash
   python3 setup_dashboard.py
   ```

3. **Launch the dashboard**
   ```bash
   python3 launch_dashboard.py
   ```

4. **Open in browser**
   - Navigate to `http://localhost:8501`

### Manual Installation
```bash
pip install streamlit plotly pandas numpy pillow requests
```

## 📁 Project Structure

```
Meta Analysis/
├── ev_ads_dashboard.py          # Main Streamlit dashboard
├── json_to_csv_converter.py     # JSON to CSV conversion
├── create_clean_csv.py          # Enhanced CSV with demographics
├── filter_car_models.py         # EV model filtering
├── identify_official_pages.py   # Page classification
├── simple_image_analyzer.py     # Basic image analysis
├── classify_ad_images.py        # AI-powered image analysis
├── setup_dashboard.py           # Dependency installer
├── launch_dashboard.py          # Dashboard launcher
├── dashboard_requirements.txt   # Python dependencies
└── README.md                    # This file
```

## 🚀 Usage

### Dashboard Navigation

1. **📊 Overview Tab**
   - Key metrics and summary statistics
   - Top car models by ad volume
   - Advertiser type distribution

2. **🚗 Car Model Analysis Tab**
   - Deep dive into specific EV models
   - Advertiser breakdown by model
   - CTA and display format analysis

3. **🎯 Ad Creative Analysis Tab**
   - Call-to-action effectiveness
   - Gender targeting patterns
   - Keyword analysis

4. **🖼️ Image Gallery Tab**
   - Visual ad browser with pagination
   - Image metadata and ad details
   - Configurable grid layout

5. **📈 Advanced Analytics Tab**
   - Time series analysis
   - Correlation matrices
   - Data export functionality

### Filtering Options

- **Car Models**: Select specific EV models to analyze
- **Advertiser Types**: Filter by official brands, dealers, or third-party advertisers
- **Date Ranges**: Analyze campaigns by time period
- **Geographic Targeting**: Focus on specific countries/regions

## 🔧 Data Processing Pipeline

1. **JSON to CSV Conversion**: Raw Facebook Ads Library JSON → Structured CSV
2. **Data Enhancement**: Add gender demographics and targeting analysis
3. **Model Filtering**: Extract ads for specific EV models
4. **Page Classification**: Identify official vs dealer vs third-party advertisers
5. **Image Analysis**: AI-powered visual content classification

## 📊 Analytics Capabilities

### Demographic Analysis
- Gender targeting breakdown (male/female percentages)
- Age group analysis
- Geographic distribution

### Creative Analysis
- Call-to-action effectiveness
- Display format preferences
- Visual theme classification
- Image content analysis

### Competitive Intelligence
- Brand vs dealer advertising strategies
- Market share by ad volume
- Targeting pattern differences

## 🎨 Image Classification

The project includes both basic and AI-powered image analysis:

### Basic Analysis
- Image dimensions and aspect ratios
- Color analysis (brightness, dominant colors)
- Basic theme classification

### AI-Powered Analysis (Optional)
- BLIP model for image captioning
- CLIP model for theme classification
- Advanced visual content understanding

## 📈 Key Insights

The dashboard reveals insights such as:
- **Market Leaders**: Volkswagen ID.4 dominates EV advertising (36.5% of ads)
- **Gender Skew**: EV ads heavily target males (85.3% vs 13.9% female)
- **Advertiser Strategy**: Most EV advertising is dealer-driven rather than brand-driven
- **Official Presence**: Only BMW maintains significant official brand presence

## 🔒 Data Privacy

- **No Data Included**: Repository contains only analysis code
- **Local Processing**: All data remains on your local machine
- **Gitignore Protection**: Comprehensive `.gitignore` prevents accidental data commits

## 🤝 Contributing

This is an analysis project. To contribute:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is for educational and research purposes. Please respect Facebook's Terms of Service when using Ads Library data.

## 🙏 Acknowledgments

- Facebook Ads Library for providing transparent advertising data
- Streamlit for the excellent dashboard framework
- Plotly for interactive visualizations
- Hugging Face Transformers for AI models

---

**Note**: This repository contains only the analysis code. Data files (CSV, JSON) are excluded via `.gitignore` and must be obtained separately from Facebook Ads Library.
