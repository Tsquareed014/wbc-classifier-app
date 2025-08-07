# WBC Classification Data Product

A Streamlit‑based web app for classifying white blood cell (WBC) images using a pretrained CNN model and visualizing results with saliency maps, confusion matrices, and batch summaries.

---

## Table of Contents

1. [Project Status](#project-status)  
2. [Features Implemented](#features-implemented)  
3. [Installation](#installation)  
4. [Usage](#usage)  
5. [Configuration](#configuration)  
6. [Deployment](#deployment)  
7. [Next Steps & To‑Do](#next-steps--to‑do)  
8. [Known Issues](#known-issues)  
9. [Contributing](#contributing)  
10.

---

## Project Status

| Component                     | Status         | Notes / Next Steps                                   |
|-------------------------------|----------------|------------------------------------------------------|
| **Image ingestion (file/URL)**| ✅ Complete    | —                                                    |
| **Exploratory analysis**      | ✅ Complete    | —                                                    |
| **Saliency visualization**    | ✅ Complete    | Fine‑tune color overlay                              |
| **Batch/ZIP processing**      | ✅ Complete    | Bulk inference performance tuning                    |
| **Export results to CSV**     | ✅ Complete    | —                                                    |
| **Adjustable text size**      | 🔲 On hold     | Add Streamlit slider and dynamic CSS injection       |
| **Keyboard navigation**       | 🔲 On hold     | Implement tabindex order and key bindings            |
| **Login / authentication**    | 🔲 Not started | Token‑based gating via `st.secrets`                  |
| **Automated tests (pytest)**  | 🔲 Partial     | Coverage ~30%; aim ≥80%                              |
| **Performance tuning**        | ✅ Complete    | Increase recall and accuracy                         |

---

## Features Implemented

- **Streamlit UI** for file/ZIP upload, batch loop, and saliency display  
- **ModelManager** loads Keras `.keras` models dynamically  
- **Preprocessing** resizes & normalizes images (128×128, mean‑std or 0–1)  
- **Saliency maps** via `tf.GradientTape` + OpenCV heatmap overlay  
- **Confusion matrix** & classification report in‑app (Scikit‑learn & Seaborn)  
- **Export to CSV** of batch predictions and metrics  
- **Memory monitoring** widget (psutil) in sidebar  

---

## Installation

1. Clone the repository:  
   ```bash
   git clone https://github.com/Tsquareed014/wbc-classifier-app.git
   cd wbc-classifier-app
   ```

2. Create & activate a virtual environment:  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:  
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Usage

```bash
streamlit run wbc_app_corrected.py   --server.port 8501   --server.headless true
```

- Open your browser at `http://localhost:8501/`.  
- Upload individual images or a ZIP of images.  
- Adjust the confidence threshold (once slider is live).  
- View saliency overlays and download CSV results.

---

## Configuration

- **Model paths** and **secrets** (e.g., API tokens) live in a `.env` file at the project root.  
- Example `.env`:
  ```
  MODEL_PATH=./models/leukocyte_classifier.keras
  ACCESS_TOKEN=your_secure_token_here
  ```

---

## Deployment

### Streamlit Community Cloud (MVP)
1. Push to GitHub.  
2. Connect your repo in [Streamlit Cloud](https://streamlit.io/cloud).  
3. Streamlit auto‑builds and hosts your app at `https://<username>.streamlit.app`.

### Self‑Hosted (Optional)
1. Provision an Ubuntu 22.04 server.  
2. Install Python 3.9, venv, Nginx, Certbot.  
3. Deploy as a `systemd` service:
   ```ini
   [Service]
   ExecStart=/path/to/venv/bin/streamlit run wbc_app_corrected.py --server.port 8501 --server.headless true
   Restart=always
   ```
4. Proxy with Nginx & secure with Let’s Encrypt.

---

## Next Steps & To‑Do

1. **Adjustable text**: Add a Streamlit slider for dynamic font sizing.  
2. **Keyboard navigation**: Ensure all controls are reachable via Tab/Enter (WCAG 2.1).  
3. **Login capability**: Implement token‑based gating using `st.secrets`.  
4. **pytest suite**: Expand to cover ingestion, preprocessing, prediction, and visualization—target ≥80% coverage.  
5. **Performance tuning**: Improve Accuracy and Recall, optimize batch handling.

---

## Known Issues

- **Saliency overlay**: Color map sometimes obscures fine details—tuning in progress.  
- **Large ZIP uploads**: Initial batch of >100 images may take >2 minutes—needs caching.  
- **No authentication**: Unauthorized users can currently view results.  

---

## Contributing

1. Fork the repo and create a feature branch.  
2. Write clear, concise code with tests.  
3. Open a pull request against `main`—at least one peer review required.

---


