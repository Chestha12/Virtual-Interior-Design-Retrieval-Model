# Virtual Interior Design Retrieval System

## Overview
The Virtual Interior Designer is an AI-powered image retrieval system that helps users find the most relevant Bedroom, Bathroom, Kitchen, Dinning, Livingroom designs based on their textual descriptions. The model takes a user input descriptions and uploaded images. The model processes user input and retrieves the top 5 most relevant images from a pretrained dataset of interior design images.

## Features
- **AI-based Image Retrieval**: Uses deep learning techniques to find the most suitable interior designs based on text input.
- **Pretrained Model with Fine-tuning**: Utilizes a combination of BLIP for image captioning, Universal Sentence Encoder (USE) for text vectorization, a CNN trained using TensorFlow, and KNN for retrieval.
- **User-friendly Interface**: Deployed using Streamlit for easy interaction.

## Technologies Used
- **Python**: Core language for model development
- **TensorFlow**: CNN model training and implementation
- **BLIP (Bootstrapped Language-Image Pretraining)**: Used for generating image descriptions
- **Universal Sentence Encoder (USE)**: Converts text descriptions into vector representations
- **K-Nearest Neighbors (KNN)**: Used for similarity-based image retrieval
- **Streamlit**: Frontend deployment
- **VS Code**: Development environment

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/virtual-interior-designer.git
   cd virtual-interior-designer
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Enter a description of your desired interior design (e.g., "modern living room with wooden flooring") or upload an image for reference.
2. The system retrieves and displays the top 5 most relevant images from the dataset based on your input.
3. Browse through the results and select your preferred design.

## Future Enhancements
- Utilize AI-driven retrieval techniques for enhanced accuracy and relevance
- Expand the dataset with a diverse range of interior designs, including different styles and layouts
- Integrate advanced deep learning models for better accuracy
- Implement a user authentication system for personalized recommendations
- Enable real-time feedback to refine search results

## Contributing
Contributions are welcome! Feel free to submit pull requests or open issues for suggestions.

