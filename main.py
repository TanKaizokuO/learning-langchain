import getpass
import os
from dotenv import load_dotenv

load_dotenv()


if os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    print("Valid NVIDIA_API_KEY already in environment. Delete to reset")
else:
    os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")


from langchain_nvidia_ai_endpoints import ChatNVIDIA

# Nemotron 3 Super — efficient reasoning and agentic tasks
llm = ChatNVIDIA(model="moonshotai/kimi-k2-instruct-0905")
result = llm.invoke("Plan a three-step research workflow for competitive analysis.")
print(result.content)
