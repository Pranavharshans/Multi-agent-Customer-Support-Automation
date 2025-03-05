import os
import warnings
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import ScrapeWebsiteTool
from mistralai import Mistral

# Load environment variables
load_dotenv()

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure Mistral Client
mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

# Define a function to get chat responses
def get_chat_response(messages):
    response = mistral_client.chat.complete(
        model="mistral-small",
        messages=messages
    )
    return response.choices[0].message.content

# Define a function to get embeddings
def get_embeddings(texts):
    response = mistral_client.embeddings.create(
        model="mistral-embed",
        inputs=texts
    )
    return [data.embedding for data in response.data]

def create_support_agents():
 
    # Support Agent
    support_agent = Agent(
        role="Senior Support Representative",
        goal="Be the most friendly and helpful support representative in your team",
        backstory=(
            "You work at crewAI (https://crewai.com) and are now working on providing "
            "support to {customer}, a super important customer for your company. "
            "You need to make sure that you provide the best support! "
            "Make sure to provide full complete answers, and make no assumptions."
        ),
        allow_delegation=False,
        verbose=True
    )

    # Support Quality Assurance Agent
    support_quality_assurance_agent = Agent(
        role="Support Quality Assurance Specialist",
        goal="Get recognition for providing the best support quality assurance in your team",
        backstory=(
            "You work at crewAI (https://crewai.com) and are now working with your team "
            "on a request from {customer} ensuring that the support representative is "
            "providing the best support possible.\n"
            "You need to make sure that the support representative is providing full "
            "complete answers, and make no assumptions."
        ),
        verbose=True,
    )

    return support_agent, support_quality_assurance_agent

def create_support_tasks(support_agent, support_quality_assurance_agent):
  
    docs_scrape_tool = ScrapeWebsiteTool(
        website_url="https://docs.crewai.com/how-to/Creating-a-Crew-and-kick-it-off/"
    )

    # Inquiry Resolution Task
    inquiry_resolution = Task(
        description=(
            "{customer} just reached out with a super important ask:\n"
            "{inquiry}\n\n"
            "{person} from {customer} is the one that reached out. "
            "Make sure to use everything you know to provide the best support possible. "
            "You must strive to provide a complete and accurate response to the customer's inquiry."
        ),
        expected_output=(
            "A detailed, informative response to the customer's inquiry that addresses "
            "all aspects of their question.\n"
            "The response should include references to everything you used to find the answer, "
            "including external data or solutions. Ensure the answer is complete, "
            "leaving no questions unanswered, and maintain a helpful and friendly "
            "tone throughout."
        ),
        tools=[docs_scrape_tool],
        agent=support_agent,
        llm=mistral_client
    )

    # Quality Assurance Review Task
    quality_assurance_review = Task(
        description=(
            "Review the response drafted by the Senior Support Representative for {customer}'s inquiry. "
            "Ensure that the answer is comprehensive, accurate, and adheres to the "
            "high-quality standards expected for customer support.\n"
            "Verify that all parts of the customer's inquiry have been addressed "
            "thoroughly, with a helpful and friendly tone.\n"
            "Check for references and sources used to find the information, "
            "ensuring the response is well-supported and leaves no questions unanswered."
        ),
        expected_output=(
            "A final, detailed, and informative response ready to be sent to the customer.\n"
            "This response should fully address the customer's inquiry, incorporating all "
            "relevant feedback and improvements.\n"
            "Don't be too formal, we are a chill and cool company "
            "but maintain a professional and friendly tone throughout."
        ),
        agent=support_quality_assurance_agent,
    )

    return inquiry_resolution, quality_assurance_review

def main():
    """
    Main function to run the multi-agent customer support automation.
    """
    # Create agents
    support_agent, support_quality_assurance_agent = create_support_agents()

    # Create tasks
    inquiry_resolution, quality_assurance_review = create_support_tasks(
        support_agent, support_quality_assurance_agent
    )

    # Create crew with memory disabled for now
    crew = Crew(
        agents=[support_agent, support_quality_assurance_agent],
        tasks=[inquiry_resolution, quality_assurance_review],
        verbose=2,
        memory=False,
        llm=mistral_client
    )

    # Define input for the crew
    inputs = {
        "customer": "DeepLearningAI",
        "person": "Andrew Ng",
        "inquiry": "I need help with setting up a Crew "
                   "and kicking it off, specifically "
                   "how can I add memory to my crew? "
                   "Can you provide guidance?"
    }

    # Run the crew and get the result
    result = crew.kickoff(inputs=inputs)

    # Print the result
    print(result)

if __name__ == "__main__":
    main()