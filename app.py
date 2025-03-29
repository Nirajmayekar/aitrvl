import gradio as gr
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

# Initialize LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key="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",
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

# Initialize conversation history
state = PlannerState(
    messages=[], city="", starting_location="", trip_duration=0, budget="", interests=[], itinerary=""
)

# Define chatbot function
# def chatbot(user_input, history):
#     global state
  
#     # Guide user step-by-step
#     if state["starting_location"] == "":
        
#         state["starting_location"] = user_input
#         return "Got it! Now, enter your **destination city** 🏙️."

#     elif state["city"] == "":
#         state["city"] = user_input
#         return "Nice! How many number of days (numerical only) are you planning to stay? 📅"

#     elif state["trip_duration"] == 0:
#         state["trip_duration"] = int(user_input)
#         return "Great! What's your **budget**? (low, mid-range, luxury) 💰"

#     elif state["budget"] == "":
#         state["budget"] = user_input
#         return "Understood! Finally, list your **interests,preferences** (e.g., beaches, museums, nightlife, dietary prefrences, travel options) 🎭."

#     elif state["interests"] == []:
#         state["interests"] = [interest.strip() for interest in user_input.split(',')]
        
#         # Generate itinerary
#         response = llm.invoke(itinerary_prompt.format_messages(
#             city=state['city'], 
#             starting_location=state['starting_location'], 
#             trip_duration=state['trip_duration'],
#             budget=state['budget'], 
#             interests=", ".join(state['interests'])
#         ))
        
#         state["itinerary"] = response.content
#         return f"✅ **Here’s your travel itinerary:**\n\n{state['itinerary']}"

#     else:
#         return "Your itinerary is ready! Do you need any modifications? ✈️"
def chatbot(user_input, history):
    # Initialize state
    if not history:
        history = {"starting_location": "", "city": "", "trip_duration": 0, "budget": "", "interests": [], "itinerary": ""}

    if history["starting_location"] == "":
        history["starting_location"] = user_input
        return "Got it! Now, enter your **destination city** 🏙️."

    elif history["city"] == "":
        history["city"] = user_input
        return "Nice! How many days are you planning to stay? 📅"

    elif history["trip_duration"] == 0:
        try:
            history["trip_duration"] = int(user_input)
        except ValueError:
            return "Please enter a **number** for trip duration (e.g., '3' instead of '3 weeks')."
        return "Great! What's your **budget**? (low, mid-range, luxury) 💰"

    elif history["budget"] == "":
        history["budget"] = user_input
        return "Understood! Finally, list your **interests** (e.g., beaches, museums, nightlife) 🎭."

    elif not history["interests"]:
        history["interests"] = [interest.strip() for interest in user_input.split(',')]
        
        # Generate itinerary
        response = llm.invoke(itinerary_prompt.format_messages(
            city=history['city'], 
            starting_location=history['starting_location'], 
            trip_duration=history['trip_duration'],
            budget=history['budget'], 
            interests=", ".join(history['interests'])
        ))

        history["itinerary"] = response.content

        return f"✅ **Here’s your travel itinerary:**\n\n{history['itinerary']}\n\nWould you like to plan another trip? 🚀"

    return "Let's start a new trip! Tell me your **starting location** again. 🌍"





# Build Gradio chatbot interface
chat_interface = gr.ChatInterface(
    chatbot,
    title="Travel Itinerary Chatbot 🤖✈️",
    description="Tell me about your trip, and I'll create a **personalized travel itinerary** for you! 🚀",
    theme="default",
    type="messages",
    examples=[["Hey, I am your Travelling planner ,tell me about your start location of your journey. "]],
)

# Launch chatbot
# chat_interface.launch()
# chat_interface.launch(server_name="0.0.0.0", server_port=8080)
import os
port = int(os.getenv("PORT", 5000))
chat_interface.launch(server_name="0.0.0.0", server_port=port)

# chat_interface.export("static")
