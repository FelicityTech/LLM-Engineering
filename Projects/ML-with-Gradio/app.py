import gradio as gr
from transformers import pipeline

# Load the pre-trained sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis")

# define the function that will wrap our model
def analyze_sentiment(text):
    result = sentiment_Analyzer(text)[0]

# define the function that will wrap our model 
def analyze_sentiment(text):
    result = sentiment_analyzer(text)[0]
    # gradi wworks best with dictionary output
    return {result['label']: result['score']}


# create the Gradio Inteface

iface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Enter a sentence here..."),
    outputs="label",
    title="Sentiment Analysis Bot",
    description="Type in a sentence and see if the model thinks it's POSITIVE or NEGATIVE. Built with Gradio and Hugging Face Transformers."
)

# launch the app!
iface.launch()