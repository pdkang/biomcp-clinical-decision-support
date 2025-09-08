# BioMCP Clinical Decision Support Extension

A powerful extension to the [BioMCP](https://github.com/genomoncology/biomcp) project with LLM that adds comprehensive clinical decision support capabilities for cancer treatment planning and research synthesis.

## 🏥 What This Project Does

This project extends the original BioMCP (Biomedical Model Context Protocol) with enhanced LLM integration capabilities that provide intelligent biomedical research and synthesis. It transforms raw biomedical data into comprehensive research reports by providing:

- **AI-Powered Research Synthesis**: Automatically generates comprehensive research reports from biomedical data
- **Intelligent Query Processing**: Natural language understanding for complex biomedical queries
- **Multi-Domain Search**: Simultaneous search across articles, clinical trials, and genomic variants
- **LLM-Enhanced Analysis**: Claude Sonnet 4 orchestration with GPT-4.1-mini synthesis
- **Rich Text Output**: Professional HTML-formatted research reports

## 🚀 Key Enhancements Over Original BioMCP

### **1. Enhanced LLM Integration**
- **Claude Sonnet 4**: Query orchestration and search planning
- **GPT-4.1-mini**: Research synthesis and comprehensive reporting
- **Intelligent Query Processing**: Automatic domain detection and parameter extraction
- **Circuit Breaker Protection**: Timeout management and error handling

### **2. Autonomous Research API**
- **Multi-Domain Search**: Simultaneous search across articles, trials, and variants
- **Intelligent Synthesis**: AI-powered research report generation
- **Rich Text Output**: HTML-formatted results for better readability
- **Configurable Analysis**: Customizable search scope and result limits

### **3. Enhanced Search Capabilities**
- **Unified Query Language**: Natural language queries across all data sources
- **Field-Specific Search**: Advanced search with structured field syntax
- **Cross-Domain Integration**: Seamless integration between different biomedical databases
- **Real-Time Data Access**: Direct access to latest research and clinical trial data


## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- API keys for Claude (Anthropic) and OpenAI

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/biomcp-clinical-decision-support.git
cd biomcp-clinical-decision-support
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
```bash
cp env.example .env
# Edit .env with your API keys:
# ANTHROPIC_API_KEY=your_anthropic_api_key_here
# BIOMCP_OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Start the Server
```bash
python start_server_simple.py
```

The server will start on `http://localhost:8000`

## 🧪 Testing the API

After starting the server with `python start_server_simple.py`, you can test the BioMCP API endpoints:

### 1. Health Check
```bash
curl http://localhost:8000/health
```

### 2. Basic BioMCP Search
```bash
# Search articles
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "BRAF melanoma",
    "domain": "article"
  }'

# Search clinical trials
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "melanoma clinical trials",
    "domain": "trial"
  }'

# Search variants
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "BRAF V600E",
    "domain": "variant"
  }'
```

### 3. Autonomous Research (LLM-Enhanced)
```bash
# Basic autonomous research
curl -X POST "http://localhost:8000/thinking/autonomous" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "BRAF V600E melanoma adjuvant therapy"
  }'

# Custom analysis scope
curl -X POST "http://localhost:8000/thinking/autonomous" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "lung cancer immunotherapy",
    "max_results_per_domain": 20,
    "samples_per_domain": 5
  }'
```

### 4. Fetch Detailed Information
```bash
# Fetch article details
curl -X POST "http://localhost:8000/fetch" \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "article",
    "id": "34567890"
  }'

# Fetch trial details
curl -X POST "http://localhost:8000/fetch" \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "trial",
    "id": "NCT04280705"
  }'
```

## 🔧 API Endpoints

### Core BioMCP Endpoints
- `GET /health` - Health check
- `POST /search` - Unified search across articles, trials, and variants
- `POST /fetch` - Fetch detailed information for specific items
- `POST /thinking/autonomous` - Autonomous research with LLM synthesis

## 🏗️ Application Platform Extensions

The following features are designed to be implemented in application platforms like Appian, not as part of the core BioMCP API:

### **Clinical Decision Support Extensions**
- **Treatment Extraction**: AI-powered parsing of research synthesis for treatment options
- **Risk-Benefit Analysis**: Comprehensive assessment of treatment risks and benefits
- **Cost Estimation**: Rule-based cost analysis for treatment options
- **Clinical Recommendations**: Actionable guidance for clinical decision making

### **Cancer Treatment Templates**
- **Standard of Care Database**: Comprehensive adjuvant therapy templates for major cancer types
- **Molecular Profile Support**: BRAF V600E melanoma, ER+ HER2- breast cancer, EGFR+ NSCLC, MSI-H colon cancer
- **Evidence-Based Protocols**: FDA-approved treatments with clinical trial references
- **Treatment Comparison**: Side-by-side analysis of multiple treatment options

### **User Interface Components**
- **Patient Dashboard**: Grid view of patients with clinical data
- **Treatment Comparison Interface**: Side-by-side treatment analysis
- **Clinical Summary Display**: Research synthesis and recommendations
- **Risk Assessment Tools**: Interactive risk-benefit analysis

### **Database Integration**
- **Patient Data Management**: Demographics, diagnoses, molecular profiles
- **Treatment History Tracking**: Surgical, radiation, clinical trial history
- **Clinical Context Storage**: ECOG scores, risk tolerance, quality of life priorities
- **Research Query Audit Trail**: Track all research queries and results

## 🧠 AI Integration Details

### **Claude Sonnet 4 (Orchestration)**
- Query parsing and intent understanding
- Search domain detection (articles, trials, variants)
- Parameter extraction from natural language
- Search planning and execution

### **GPT-4.1-mini (Synthesis)**
- Research synthesis and report generation
- Comprehensive analysis of search results
- Structured report formatting
- Evidence quality assessment

## 📊 Key Features

### **Autonomous Research**
- Multi-domain search across articles, trials, and variants
- Intelligent query optimization for better results
- Comprehensive research synthesis
- Rich HTML output for web applications

### **Enhanced Search**
- Natural language query processing
- Field-specific search capabilities
- Cross-domain data integration
- Real-time biomedical data access

### **LLM-Powered Analysis**
- Intelligent query understanding
- Automated search domain detection
- Comprehensive research synthesis
- Professional report generation

## 🔬 Use Cases

1. **Researchers**: Comprehensive biomedical research and literature review
2. **Clinicians**: Evidence-based treatment research and analysis
3. **Healthcare Systems**: Integrated biomedical data access
4. **Medical Students**: Learning and research support
5. **Application Developers**: Building clinical decision support systems

## 🤝 Contributing

This project extends the original BioMCP with enhanced LLM integration capabilities. Contributions are welcome for:

- Enhanced AI prompts for research synthesis
- New biomedical data source integrations
- Performance optimizations for LLM processing
- Additional search capabilities
- Improved query processing algorithms

## 📄 License

This project is licensed under the MIT License, same as the original BioMCP project.
  
## 🙏 Acknowledgments

- **Original BioMCP**: Built upon the excellent foundation provided by [GenomOncology's BioMCP](https://github.com/genomoncology/biomcp)
- **Claude & GPT**: Powered by Anthropic's Claude Sonnet 4 and OpenAI's GPT-4.1-mini
- **Biomedical Data Sources**: Integrates with PubMed, ClinicalTrials.gov, MyVariant.info, and cBioPortal

  
---

**Note**: This project is designed for research and educational purposes. For clinical use, ensure compliance with healthcare regulations and validate all recommendations with qualified medical professionals.
