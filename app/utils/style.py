import streamlit as st


def footer():
    st.markdown(
        """
        <style>
            .footer-container {
                margin-top: 60px;
                padding: 30px 20px;
                background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
                border-top: 2px solid #b6dce5;
                border-radius: 10px 10px 0 0;
                box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.05);
            }
            
            .footer-content {
                max-width: 1200px;
                margin: 0 auto;
                text-align: center;
            }
            
            .footer-text {
                color: #1f2937;
                font-size: 14px;
                font-weight: 500;
                margin-bottom: 20px;
                font-family: system-ui, -apple-system, sans-serif;
            }
            
            .footer-links {
                display: flex;
                justify-content: center;
                align-items: center;
                gap: 20px;
                flex-wrap: wrap;
            }
            
            .footer-link {
                display: inline-block;
                transition: transform 0.3s ease, opacity 0.3s ease;
                opacity: 0.8;
            }
            
            .footer-link:hover {
                transform: scale(1.1);
                opacity: 1;
            }
            
            .footer-link img {
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
                transition: box-shadow 0.3s ease;
            }
            
            .footer-link:hover img {
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            }
        </style>

        <div class="footer-container">
            <div class="footer-content">
                <p class="footer-text">© 2025 Syed Muhammad Rafay. All rights reserved.</p>
                <div class="footer-links">
                    <a href="https://colab.research.google.com/github/Syed-Muhammad-Rafay/ID-Fan-Project-MS-Thesis-/blob/main/notebooks/main.ipynb" 
                       class="footer-link" 
                       target="_blank" 
                       rel="noopener noreferrer">
                        <img src="https://colab.research.google.com/assets/colab-badge.svg" 
                             alt="Open in Colab" 
                             width="90" 
                             height="50">
                    </a>
                    <a href="https://github.com/Syed-Muhammad-Rafay/ID-Fan-Project-MS-Thesis-" 
                       class="footer-link" 
                       target="_blank" 
                       rel="noopener noreferrer">
                        <img src="https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png" 
                             alt="GitHub Repository" 
                             width="55" 
                             height="55">
                    </a>
                </div>
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )
