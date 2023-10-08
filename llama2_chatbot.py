import transformers
import torch
from transformers import AutoTokenizer
import gradio
from dotenv import load_dotenv
from huggingface_hub import login
import os


api_token =  os.getenv('HUGGINGFACEHUB_API_TOKEN')

login(token=api_token)

model = 'meta-llama/Llama-2-7b-chat-hf'
tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token = True)  ##To the format in to the one in which the model can understand that.

##Now creating the pipeline for our model

from transformers import pipeline
llama_pipeline = pipeline(
    'text-generation',
    model = model,
    torch_dtype = torch.float16,
    device_map = 'auto'
)

##Now a basic function for getting the response from the llm to our query. We simply pass the prompt/question/query and let it generate the text.

def get_response(prompt: str) -> None:

  sequences = llama_pipeline(
      prompt,   ##We pass in the prompt to our Pipeline
      do_sample = True,  ##We keep sampling to True
      top_k = 10,  ##Here we define the number of top searches
      num_return_sequences = 1,   ##Here we define the number of returned sequences
      eos_token_id = tokenizer.eos_token_id, ##This is the eos token id
      max_length = 256  ##This is the length of the returning sequence or the response from the llm.
  )

  print("Chatbot: ", sequences[0]['generated_text'])

##Now let's check our model's performance.

#get_response("My name is Piyush.")

#get_response("I am persuing to be an LLM Engineer right now.")

#get_response("What's I am persuing to be right now?")

#get_response("What is my name?")

##As you can see, LLM is generating just some trash responses. That is the limitation of the function we just wrote.

##We need to tweak that function a little but to make the model answer or respond to our query more accurately.

##It does not have a memory right now.
##We can not customize it either by passing the prompts.
##And it is also not ready to be integrated on the User Interface Platforms.

##These are the drawback of the function we just defined.

##Now to improve its functioning. We can right a prompts to give it instructions and also the make memory where we can store our chat history.
##And also format our text or message or input.

##Now the prompt to llama2 is provided in this way:-

###<s> [INST] <<SYS>>
###{{ system_prompt }}                                                     ##We enter the 'system prompt' between the '<<SYS>>' tag
###<</SYS>>  {{ user_message }} [\INST]  {{ bot_message/response }} </s>   ##we enter the 'user_message' between the '[INST]' tags and the 'bot_response' between the '[/INST]' and '</s>' tags.

###This is the format for providing the prompts to our llama.

##Let's build the Prompt

SYSTEM_PROMPT = """<s>[INST] <<SYS>>
You are a helpful bot. YOur responses are Concise.
<</SYS>>
"""

##We are gonna use this prompt to format our message.

##So let's write a function to format our message and control the chat_history.
##Basically 'history/chat_history' is a list of tuple with 'user_msg' and 'bot_msg'.
##i.e.  [(user_msg, bot_msg), (user_msg, bot_msg).....]

def format_message(message: str, history: list, memory_limit: int = 3) -> str:

  ##This function formats the message and the history for our llama model. And it returns the formatted message string.
  ##'message' is the current message that user sends.
  ##'history' is the list like mentioned above.
  ##'memory_limit' is the limit of the list upto how many it can save within itself.   'len(history)<=memory_limit'

  ##We are gonna 'Gradio's Chat_Interface module' that has the interface 'history' and 'message->user's last message'.

  if len(history) > memory_limit:
    history = history[-memory_limit:]   ##It slices the history iterable, keeping only the last memory_limit elements and assigns the result back to the variable history.

  if len(history) == 0:
    return SYSTEM_PROMPT + f"{message} [/INST]"   ##Here we added the 'user message' i.e. 'message' parameter to our system prompt, that we wrote upto '<</SYS>>' tag. So we basically added the user's message into that.

  formatted_message = SYSTEM_PROMPT + f"{history[0][0]} [/INST] {history[0][1]} </s>"  ##Here we added more to/completed our prompt and assigned it to our formatted message.
                                                                                       ##'history[0][0]' is the user msg and the 'history[0][1]' is the bot response.
  ##When our contect length grows we store the whole conversation in the history.
  ##Handle the coversation history
  for user_msg, bot_response in history[1:]:  ##we started from the 1st index onwards because we have already define the formatted message for our history's 0th index above.
    formatted_message += f"<s>[INST] {user_msg} [/INST] {bot_response} </s>"   ##We keep on adding to our format message the user_msg and the bot_response.

  ##Handling our last message
  formatted_message += f"<s>[INST] {message} [/INST]"   ##Here we handled our last message.


  return formatted_message

  ##So basically, in the 0th index or the first time in our formatted message, we added the system prompt and the user_message.
  ##then we added more to our formatted text, the user message and the bot response.
  ##Then after that, from the 1st index, we keep on adding to our formatted text the user_msg and the bot_response from the history.
  ##At last we have our own message that we want to add to our formatted text. Now we do not have any bot message with it. So we just add user_msg to the formatted_text.

  ##This 'formatted_text' will be what we give in as our 'query/message/question' to our model.
  ##We will always send the whole history when we prompt or query our llama2 now. That's how we handle this open-source model. That's why 'history' is so useful here.

##Now this function only gives our formatted text that we will give to our model.
##We need to define one more function to get the response from our LLM.

def get_llm_response(message: str, history: list) -> str:

  query = format_message(message, history)   ##This is where we format our query to later give it to our LLM.
  response = ""

  sequences = llama_pipeline(
      query,
      do_sample = True,
      top_k = 10,
      num_return_sequences = 1,
      eos_token_id = tokenizer.eos_token_id,
      max_length = 1024
  )

  generated_text = sequences[0]['generated_text']
  response = generated_text[len(query):]  ##Here we are removing the query that we pass onto our llm.
  print("Chatbot: ", response.strip())
  return response.strip()

##Now let's use gradio for a chat interface.
##We will use the 'chat_interface' module of gradio.


gradio.ChatInterface(get_llm_response).launch()  ##We just need to pass in the main function where we are returning response from the llm. And gradio will take care of everything.

