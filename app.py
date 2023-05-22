import gradio as gr
from sentence_transformers import SentenceTransformer, util
model_sentence = SentenceTransformer('all-MiniLM-L6-v2')

para_1 ="""
Natural language processing (NLP) is a field of computer science that studies how computers can understand and process human language. NLP is a subfield of artificial intelligence (AI) that deals with the interaction between computers and human (natural) languages.

NLP has a wide range of applications, including:

Machine translation: translating text from one language to another
Text summarization: extracting the main points of a text
Question answering: answering questions posed in natural language
Text classification: classifying text into categories, such as spam or ham
Sentiment analysis: determining the sentiment of a text, such as positive, negative, or neutral
Natural language generation: generating text that is similar to human-written text
NLP is a challenging field, as human language is complex and nuanced. However, NLP has made significant progress in recent years, and it is now a powerful tool that can be used to solve a wide range of problems.


"""
para_2 ="""
Generative adversarial networks (GANs) are a type of machine learning model that can be used to generate realistic and creative content. GANs were first introduced in 2014 by Ian Goodfellow, and they have since been used to generate a wide range of content, including images, text, and music.

GANs work by pitting two neural networks against each other in a game-like setting. One network, the generator, is responsible for creating new content. The other network, the discriminator, is responsible for determining whether the content created by the generator is real or fake.

The generator is trained to create content that is as realistic as possible, while the discriminator is trained to distinguish between real and fake content. As the two networks compete against each other, they both become better at their respective tasks.

GANs have been used to generate a wide range of content, including:

Images: GANs have been used to generate realistic images of people, animals, and objects.
Text: GANs have been used to generate realistic text, such as news articles, blog posts, and even poetry.
Music: GANs have been used to generate realistic music, such as songs, symphonies, and even jazz improvisations.
GANs are a powerful tool that can be used to generate realistic and creative content. As GANs continue to develop, they are likely to be used to create even more amazing and impressive content in the future.


"""
def paragraph_similar(text1, text2):
    sentences = []
    sentences.append(text1)
    sentences.append(text2)
    paraphrases = util.paraphrase_mining(model_sentence, sentences, corpus_chunk_size=len(sentences))
    return {"Similarity": [round(paraphrases[0][0], 2)]}

 
with gr.Blocks(title="Paragraph",css="footer {visibility: hidden}") as demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Paragraph Compare")
            with gr.Row():           
                with gr.Column():
                    inputs_1 = gr.TextArea(label="Paragraph 1",value=para_1,interactive=True)
                    inputs_2 = gr.TextArea(label="Paragraph 2",value=para_2,interactive=True)
                with gr.Column():
                    btn = gr.Button(value="RUN")
                    output = gr.Label(label="output")
                btn.click(fn=paragraph_similar,inputs=[inputs_1,inputs_2],outputs=[output])
demo.launch()