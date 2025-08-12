// Gene-Disease Analysis Service - Frontend Application

class GeneDiseasApp {
    constructor() {
        this.sessionId = localStorage.getItem('session_id');
        this.username = localStorage.getItem('username');
        this.apiProvider = localStorage.getItem('api_provider');
        this.ws = null;
        this.currentAnalysisId = null;
        
        this.init();
    }
    
    clearSession() {
        this.sessionId = null;
        this.username = null;
        this.apiProvider = null;
        
        localStorage.removeItem('session_id');
        localStorage.removeItem('username');
        localStorage.removeItem('api_provider');
    }
    
    async init() {
        // Check if we have stored session data
        if (this.sessionId && this.username) {
            // Validate session with backend
            try {
                const response = await fetch(`/api/v1/analyses?session_id=${this.sessionId}&limit=1`);
                if (response.ok) {
                    // Session is valid, stay logged in
                    this.showLoggedInUI();
                } else {
                    // Session expired or invalid, clear and show login
                    this.clearSession();
                    this.hideLoggedInUI();
                }
            } catch (error) {
                // Network error or backend down, clear session and show login
                this.clearSession();
                this.hideLoggedInUI();
            }
        } else {
            // No stored session, show login page
            this.hideLoggedInUI();
        }
        
        // Bind event handlers
        document.getElementById('login-form').addEventListener('submit', (e) => this.handleLogin(e));
        document.getElementById('logout-btn').addEventListener('click', () => this.handleLogout());
        document.getElementById('analysis-form').addEventListener('submit', (e) => this.handleAnalysis(e));
        document.getElementById('refresh-history').addEventListener('click', () => this.loadHistory());
        
        // Update model options based on provider
        document.getElementById('api-provider').addEventListener('change', (e) => this.updateModelOptions(e.target.value));
    }
    
    showLoggedInUI() {
        document.getElementById('login-section').style.display = 'none';
        document.getElementById('analysis-section').style.display = 'block';
        document.getElementById('history-section').style.display = 'block';
        document.getElementById('user-info').style.display = 'flex';
        document.getElementById('username-display').textContent = this.username;
        
        this.updateModelOptions(this.apiProvider);
        this.loadHistory();
    }
    
    hideLoggedInUI() {
        document.getElementById('login-section').style.display = 'block';
        document.getElementById('analysis-section').style.display = 'none';
        document.getElementById('history-section').style.display = 'none';
        document.getElementById('user-info').style.display = 'none';
    }
    
    updateModelOptions(provider) {
        const modelSelect = document.getElementById('model');
        modelSelect.innerHTML = '';
        
        if (provider === 'openai') {
            modelSelect.innerHTML = `
                <option value="gpt-5-mini" selected>GPT-5 Mini</option>
                <option value="gpt-5">GPT-5</option>
                <option value="gpt-5-nano">GPT-5 Nano</option>
            `;
        } else if (provider === 'anthropic') {
            modelSelect.innerHTML = `
                <option value="claude-sonnet-4-20250514">Claude 4 Sonnet</option>
                <option value="claude-opus-4-20250514">Claude 4 Opus</option>
            `;
        }
    }
    
    async handleLogin(e) {
        e.preventDefault();
        
        const formData = new FormData(e.target);
        const data = {
            username: formData.get('username'),
            api_provider: formData.get('api_provider'),
            api_key: formData.get('api_key')
        };
        
        try {
            const response = await fetch('/api/v1/auth/login', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(data)
            });
            
            if (!response.ok) {
                throw new Error('Login failed');
            }
            
            const result = await response.json();
            
            // Store session info
            this.sessionId = result.session_id;
            this.username = result.username;
            this.apiProvider = data.api_provider;
            
            localStorage.setItem('session_id', this.sessionId);
            localStorage.setItem('username', this.username);
            localStorage.setItem('api_provider', this.apiProvider);
            
            this.showLoggedInUI();
            
            // Clear form
            e.target.reset();
            
        } catch (error) {
            alert('Login failed: ' + error.message);
        }
    }
    
    handleLogout() {
        this.clearSession();
        this.hideLoggedInUI();
    }
    
    async handleAnalysis(e) {
        e.preventDefault();
        
        const formData = new FormData(e.target);
        const data = {
            gene: formData.get('gene'),
            disease: formData.get('disease'),
            since_year: parseInt(formData.get('since_year')),
            max_abstracts: parseInt(formData.get('max_abstracts')),
            include_gwas: formData.get('include_gwas') === 'on',
            model: formData.get('model')
        };
        
        // Show status
        document.getElementById('analysis-status').style.display = 'block';
        document.getElementById('analysis-results').style.display = 'none';
        document.getElementById('status-text').textContent = 'Starting analysis...';
        document.getElementById('status-spinner').style.display = 'block';
        
        try {
            const response = await fetch('/api/v1/analyses?session_id=' + this.sessionId, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(data)
            });
            
            if (!response.ok) {
                throw new Error('Failed to start analysis');
            }
            
            const result = await response.json();
            this.currentAnalysisId = result.id;
            
            // Connect WebSocket for real-time updates
            this.connectWebSocket(result.id);
            
            // Start polling for results
            this.pollForResults(result.id);
            
        } catch (error) {
            alert('Analysis failed: ' + error.message);
            document.getElementById('analysis-status').style.display = 'none';
        }
    }
    
    connectWebSocket(analysisId) {
        if (this.ws) {
            this.ws.close();
        }
        
        // Use wss:// for HTTPS and ws:// for HTTP
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}/api/v1/ws/${analysisId}`;
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            document.getElementById('status-text').textContent = data.message || 'Processing...';
            
            if (data.status === 'completed' || data.status === 'failed') {
                document.getElementById('status-spinner').style.display = 'none';
                this.loadAnalysisResults(analysisId);
            }
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }
    
    async pollForResults(analysisId) {
        let attempts = 0;
        const maxAttempts = 60; // 60 seconds timeout
        
        const poll = async () => {
            if (attempts >= maxAttempts) {
                document.getElementById('status-text').textContent = 'Analysis timeout';
                document.getElementById('status-spinner').style.display = 'none';
                return;
            }
            
            try {
                const response = await fetch(`/api/v1/analyses/${analysisId}?session_id=${this.sessionId}`);
                const result = await response.json();
                
                if (result.status === 'completed' || result.status === 'failed') {
                    this.displayResults(result);
                    this.loadHistory(); // Refresh history
                } else {
                    attempts++;
                    setTimeout(poll, 1000);
                }
            } catch (error) {
                console.error('Polling error:', error);
                attempts++;
                setTimeout(poll, 1000);
            }
        };
        
        setTimeout(poll, 2000); // Start polling after 2 seconds
    }
    
    async loadAnalysisResults(analysisId) {
        try {
            const response = await fetch(`/api/v1/analyses/${analysisId}?session_id=${this.sessionId}`);
            const result = await response.json();
            this.displayResults(result);
            this.loadHistory(); // Refresh history
        } catch (error) {
            console.error('Failed to load results:', error);
        }
    }
    
    displayResults(result) {
        document.getElementById('analysis-status').style.display = 'none';
        document.getElementById('analysis-results').style.display = 'block';
        
        const resultsContent = document.getElementById('results-content');
        
        if (result.status === 'failed' || result.error_message) {
            resultsContent.innerHTML = `
                <div class="error-message">
                    <strong>Analysis Failed:</strong> ${result.error_message || 'Unknown error'}
                </div>
            `;
            return;
        }
        
        let html = '<div class="result-section">';
        html += `<h4>Gene: ${result.gene_symbol} → Disease: ${result.disease_label}</h4>`;
        
        if (result.verdict) {
            html += `<div class="verdict ${result.verdict}">${result.verdict.toUpperCase()}</div>`;
        }
        
        if (result.confidence !== null && result.confidence !== undefined) {
            html += `
                <div>Confidence: ${(result.confidence * 100).toFixed(1)}%</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${result.confidence * 100}%"></div>
                </div>
            `;
        }
        
        html += '</div>';
        
        // Display LLM output if available
        if (result.llm_output) {
            html += '<div class="result-section"><h4>Analysis Details</h4>';
            
            // Key points
            if (result.llm_output.key_points && result.llm_output.key_points.length > 0) {
                html += '<h5>Key Findings:</h5>';
                result.llm_output.key_points.forEach(point => {
                    html += `<div class="key-point">${point.statement}</div>`;
                });
            }
            
            // Drivers
            if (result.llm_output.drivers) {
                html += '<h5>Evidence Drivers:</h5><ul>';
                for (const [key, driver] of Object.entries(result.llm_output.drivers)) {
                    if (driver.present) {
                        html += `<li><strong>${key}:</strong> ${driver.summary}</li>`;
                    }
                }
                html += '</ul>';
            }
            
            
            html += '</div>';
        }
        
        // Display evidence summary
        if (result.evidence) {
            html += '<div class="result-section"><h4>Evidence Summary</h4>';
            
            // Open Targets
            if (result.evidence.opentargets) {
                const ot = result.evidence.opentargets;
                html += `
                    <div class="evidence-item">
                        <h5>Open Targets Platform</h5>
                        <p>Overall Association Score: ${(ot.overall_association_score || 0).toFixed(3)}</p>
                    </div>
                `;
            }
            
            // Literature
            if (result.evidence.literature && result.evidence.literature.length > 0) {
                html += `
                    <div class="evidence-item">
                        <h5>Literature Evidence</h5>
                        <p>Found ${result.evidence.literature.length} relevant publications:</p>
                        <div class="literature-links">
                `;
                
                result.evidence.literature.forEach((pub, index) => {
                    const title = pub.title || 'Untitled';
                    const uri = pub.uri || '';
                    const pmid = pub.pmid || '';
                    const year = pub.year || '';
                    
                    html += `
                        <div class="lit-link">
                            ${uri ? 
                                `<a href="${uri}" target="_blank" rel="noopener">${title}</a>` : 
                                title}
                            ${year ? ` (${year})` : ''}
                            ${pmid ? ` <span class="pmid">PMID: ${pmid}</span>` : ''}
                        </div>
                    `;
                });
                
                html += `
                        </div>
                    </div>
                `;
            }
            
            // GWAS
            if (result.evidence.gwas_catalog && result.evidence.gwas_catalog.length > 0) {
                html += `
                    <div class="evidence-item">
                        <h5>GWAS Catalog</h5>
                        <p>Found ${result.evidence.gwas_catalog.length} genetic associations</p>
                    </div>
                `;
            }
            
            html += '</div>';
        }
        
        resultsContent.innerHTML = html;
    }
    
    async loadHistory() {
        try {
            const response = await fetch(`/api/v1/analyses?session_id=${this.sessionId}`);
            const history = await response.json();
            
            const historyList = document.getElementById('history-list');
            
            if (history.length === 0) {
                historyList.innerHTML = '<p>No analyses yet</p>';
                return;
            }
            
            let html = '';
            history.forEach(item => {
                const date = new Date(item.created_at).toLocaleString();
                html += `
                    <div class="history-item" onclick="app.loadAnalysisResults(${item.id})">
                        <div class="history-header">
                            <strong>${item.gene_symbol} → ${item.disease_label}</strong>
                            <span class="verdict ${item.verdict || ''}">${item.verdict || item.status}</span>
                        </div>
                        <div class="history-meta">
                            <span>ID: ${item.id}</span>
                            <span>${date}</span>
                            ${item.confidence !== null && item.confidence !== undefined ? 
                                `<span>Confidence: ${(item.confidence * 100).toFixed(1)}%</span>` : ''}
                        </div>
                    </div>
                `;
            });
            
            historyList.innerHTML = html;
            
        } catch (error) {
            console.error('Failed to load history:', error);
        }
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new GeneDiseasApp();
});