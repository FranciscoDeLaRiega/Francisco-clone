import os
import uvicorn
from loguru import logger
from francisco_agent import CloneExecutor
from a2a.types import AgentCard, AgentCapabilities, AgentInterface, AgentSkill
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.server.apps import A2AStarletteApplication
from a2a.server.tasks import InMemoryTaskStore, InMemoryPushNotificationConfigStore
from a2a.server.events import InMemoryQueueManager
from starlette.middleware.cors import CORSMiddleware


PORT = 9000
AGENT_CARD = AgentCard(
    name="Francisco's Clone",
    description=("A multi-skill agent that can perform: "
    "1-LLM-style question answering: elementary-level math problems " 
    "2-Tool use: execute a sequence of sha512 and md5 operations " 
    "3-Image understanding: select what is in the image "
    "4-Web browsing: win Tic-tac-toe "
    "5-Code generation and execution: brute-force algorithm implementation "
    "6-Memorizing tasks across sessions"),
    version="1.0.0",
    url=f"http://localhost:{PORT}",
    preferred_transport="http",
    capabilities=AgentCapabilities(
        streaming=True,
        completion=True,
        function_calling=False,
        sampling=True
    ),
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
    skills=[
        AgentSkill(
            id="math",
            name="Elementary Math",
            description="Solve elementary-level math problems",
            tags=["math", "arithmetic", "geometry", "problem-solving", "education"],
            input_modes=["text/plain"],
            output_modes=["text/plain"]
        ),

        AgentSkill(
            id="hashing_pipeline",
            name="Hash Chaining (SHA-512 & MD5)",
            description="Perform a chain of hashing operations using SHA-512 and MD5, returning intermediate and final digests.",
            tags=["hashing", "sha512", "md5", "cryptography", "tools"],
            input_modes=["text/plain"],
            output_modes=["text/plain"]
        ),

        AgentSkill(
            id="image_analysis",
            name="Image Analysis & Classification",
            description="Inspect images, describe their content, and answer related questions.",
            tags=["vision", "image-recognition", "image-classification", "analysis", "multimodal"],
            input_modes=["text/plain", "image/*"],
            output_modes=["text/plain"]
        ),

        AgentSkill(
            id="web_agent",
            name="Automated Web Tasks",
            description="Navigate the web, retrieve information, and complete interactive tasks (e.g., Tic-tac-toe).",
            tags=["web", "automation", "browsing", "search", "information-retrieval"],
            input_modes=["text/plain"],
            output_modes=["text/plain"]
        ),

        AgentSkill(
            id="code_runner",
            name="Code Generation & Execution",
            description="Generate and execute code (e.g., brute-force algorithms) and report results.",
            tags=["programming", "code-execution", "algorithms", "automation", "developer-tools"],
            input_modes=["text/plain"],
            output_modes=["text/plain"]
        ),

        AgentSkill(
            id="memory_manager",
            name="Contextual Memory",
            description="Store and recall user preferences, facts, and goals across sessions; summarize context for next steps.",
            tags=["memory", "context", "long-term-memory", "personalization", "stateful"],
            input_modes=["text/plain"],
            output_modes=["text/plain"]
        ),
    ],
    additional_interfaces=[
        AgentInterface(
            url=f"http://localhost:{PORT}",
            transport="http"
        )
    ]
)
def create_app():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required")

    app = A2AStarletteApplication(
        agent_card=AGENT_CARD,
        http_handler=DefaultRequestHandler(
            agent_executor=CloneExecutor(api_key),
            task_store=InMemoryTaskStore(),
            push_config_store=InMemoryPushNotificationConfigStore(),
            queue_manager=InMemoryQueueManager(),
        ),
    ).build()

    # CORS: open for dev; restrict in prod
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], 
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    logger.info("A2A agent ready on port %s", PORT)
    return app


if __name__ == "__main__":
    uvicorn.run("FranciscoClone.__main__:create_app", factory=True, host="0.0.0.0", port=PORT)

