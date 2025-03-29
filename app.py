import gradio as gr
import os
from typing import TypedDict, Annotated, List
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Define planner state
class PlannerState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], "Conversation history"]
    city: str
    starting_location: str
    trip_duration: int
    budget: str
    interests: List[str]
    itinerary: str

groq_api_key = os.getenv("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")
if not groq_api_key:
    raise ValueError("🚨 Error: Missing GROQ_API_KEY. Set it in environment variables.")

# Initialize LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key,  # Secure your API key
    model_name="llama-3.3-70b-versatile"
)

# Define prompts
itinerary_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a travel assistant.  
    Generate a **{trip_duration}-day itinerary** for {city}, considering:
    - **Starting Location**: {starting_location}  
    - **Interests**: {interests}  
    - **Budget**: {budget}  

    **Include realistic travel plans** from {starting_location} to {city} (e.g., flight, train, road trip).  
    **Ensure travel time is factored into the schedule.**  
    **Format the response with specific time slots**, like this:

    **Day 1**  
    - 06:00 AM - Depart from {starting_location} via [Flight/Train/Car] ✈️  
    - 09:00 AM - Arrive in {city} and check in to your accommodation 🏨  
    - 10:30 AM - Breakfast at [Affordable/Luxury] restaurant ☕  
    - 12:00 PM - Visit [Main Attraction] 🏛️  
    - 02:30 PM - Lunch at [Budget-friendly/Luxury] spot 🍽️  
    - 04:00 PM - Explore [Another Activity] 🎭  
    - 07:30 PM - Dinner at [Affordable/Luxury] restaurant 🍷  
    - 09:00 PM - Evening entertainment 🎶  
    """),
    ("human", "Create a detailed, time-based itinerary for my trip."),
])

# Initialize state
def init_state():
    return PlannerState(messages=[], city="", starting_location="", trip_duration=0, budget="", interests=[], itinerary="")

# Define chatbot function
def chatbot(user_input, history, state):
    if not state:
        state = init_state()

    print("STATE BEFORE:", state)  # Debugging step

    # Guide user step-by-step
    if state["starting_location"] == "":
        state["starting_location"] = user_input
        return "Got it! Now, enter your **destination city** 🏙️.", state

    elif state["city"] == "":
        state["city"] = user_input
        return "Nice! How many days are you planning to stay? 📅", state

    elif state["trip_duration"] == 0:
        try:
            state["trip_duration"] = int(user_input)
            return "Great! What's your **budget**? (low, mid-range, luxury) 💰", state
        except ValueError:
            return "⚠️ Please enter a **valid number** for trip duration (e.g., 3, 7, 14).", state

    elif state["budget"] == "":
        state["budget"] = user_input
        return "Understood! Finally, list your **interests** (e.g., beaches, museums, nightlife) 🎭.", state

    elif not state["interests"]:
        state["interests"] = [interest.strip() for interest in user_input.split(',')]

        # Generate itinerary
        response = llm.invoke(itinerary_prompt.format_messages(
            city=state['city'], 
            starting_location=state['starting_location'], 
            trip_duration=state['trip_duration'],
            budget=state['budget'], 
            interests=", ".join(state['interests'])
        ))

        state["itinerary"] = response.content
        state = init_state()
        return f"✅ **Here’s your travel itinerary:**\n\n{state['itinerary']}", state

    else:
        state = init_state()
        return "Your itinerary is ready! Do you need any modifications? ✈️", state

# Use Gradio Blocks instead of ChatInterface
with gr.Blocks() as demo:
    chatbot_ui = gr.Chatbot()
    user_input = gr.Textbox(placeholder="Type your message here...")
    submit_button = gr.Button("Submit")
    state = gr.State(init_state())  # Add persistent state

    def process_input(user_input, chat_history, state):
        response, state = chatbot(user_input, chat_history, state)
        chat_history.append((user_input, response))
        return chat_history, state

    submit_button.click(process_input, inputs=[user_input, chatbot_ui, state], outputs=[chatbot_ui, state])

# Launch chatbot
port = int(os.getenv("PORT", 5000))
demo.launch(server_name="0.0.0.0", server_port=port)
