import os
import asyncio
import logging
import json
import base64
import threading
from queue import Queue, Empty
from io import BytesIO
from PIL import Image

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from gsima import environments
from gsima.utils.logging import setup_logging
from gsima.utils import config
from gsima import runtime as runtime_factory
from gsima.agents import create_agent
from gsima.agents.context import AgentContext

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

setup_logging()

# Global state to track active run
active_agent_thread = None
agent_stop_event = threading.Event()
message_queue = Queue()


def _get_initial_frame_sync(q: Queue):
    """Fetches the first frame of the environment without running the agent."""
    import random
    original_render_mode = config.RENDER_MODE
    try:
        # Prevent creating empty video recording folders just for the UI preview frame
        if config.RENDER_MODE == "record":
            config.RENDER_MODE = "rgb_array"
            
        # Generate a seed so the agent loop will use the identical layout
        config.ENV_SEED = random.randint(0, 999999)
            
        env, _, _, _ = environments.create_env_and_adapter()
        env.reset(seed=config.ENV_SEED)
        if config.RENDER_MODE in ("human", "record"):
            rgb = env.unwrapped.render()
        else:
            rgb = env.render()
            
        img = Image.fromarray(rgb)
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=80)
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        q.put({"type": "frame", "data": b64})
    except Exception as e:
        logging.error(f"Failed to get initial frame: {e}")
    finally:
        config.RENDER_MODE = original_render_mode
        if 'env' in locals() and env:
            env.close()

def _run_agent_sync(instruction: str, q: Queue, stop_event: threading.Event):
    """Synchronous thread that runs the agent loop."""
    config.INSTRUCTION = instruction
    
    try:
        multimodal_runtime = runtime_factory.create_runtime(config.RUNTIME, "multimodal", config.GEMINI_MODEL)
        env, adapter, memory_system, get_multimodal_prompt = environments.create_env_and_adapter()
        
        def event_emitter(event_type: str, data: any):
            if stop_event.is_set():
                # We can't gracefully exit the environment step easily without throwing,
                # but we can stop emitting and let it finish.
                return
                
            if event_type == "frame":
                try:
                    img = Image.fromarray(data)
                    buffer = BytesIO()
                    img.save(buffer, format="JPEG", quality=80)
                    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    q.put({"type": "frame", "data": b64})
                except Exception as e:
                    logging.error(f"Failed to encode frame: {e}")
                    
            elif event_type == "thought":
                q.put({"type": "thought", "data": data})

        context = AgentContext(
            env=env,
            adapter=adapter,
            multimodal_runtime=multimodal_runtime,
            memory_system=memory_system,
            get_multimodal_prompt=get_multimodal_prompt,
            event_emitter=event_emitter
        )

        agent = create_agent()
        logging.info("Starting agent run from API...")
        
        eval_result = agent.run(context)
        q.put({"type": "status", "data": "finished"})
        
    except Exception as e:
        logging.error(f"Agent thread error: {e}")
        q.put({"type": "error", "data": str(e)})
    finally:
        if 'env' in locals() and env:
            env.close()

async def ws_message_pump(websocket: WebSocket):
    """Pumps messages from the queue to the websocket."""
    while True:
        try:
            # Non-blocking check for messages
            msg = message_queue.get_nowait()
            await websocket.send_json(msg)
        except Empty:
            await asyncio.sleep(0.01) # Yield to event loop
        except asyncio.CancelledError:
            break
        except RuntimeError as e:
            if "Unexpected ASGI message" in str(e):
                break
            logging.error(f"WebSocket pump error: {e}")
            break
        except Exception as e:
            logging.error(f"WebSocket pump error: {e}")
            break

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global active_agent_thread, agent_stop_event
    await websocket.accept()
    
    # Reset the seed for a fresh layout on page reload
    config.ENV_SEED = None
    
    try:
        # Send initial config state to UI
        await websocket.send_json({
            "type": "init",
            "env_name": config.UI_GAME_NAME,
            "default_instruction": config.INSTRUCTION
        })
        
        pump_task = None
        # Start the message pump loop exactly once per connection
        pump_task = asyncio.create_task(ws_message_pump(websocket))
        
        # Fetch initial frame in the background
        threading.Thread(target=_get_initial_frame_sync, args=(message_queue,), daemon=True).start()

        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            
            if payload.get("action") == "start":
                instruction = payload.get("instruction", "Solve the task.")
                
                # Stop any existing run
                if active_agent_thread and active_agent_thread.is_alive():
                    agent_stop_event.set()
                    active_agent_thread.join(timeout=2.0)
                
                # Clear queue
                while not message_queue.empty():
                    try:
                        message_queue.get_nowait()
                    except Empty:
                        break
                        
                agent_stop_event.clear()
                active_agent_thread = threading.Thread(
                    target=_run_agent_sync,
                    args=(instruction, message_queue, agent_stop_event),
                    daemon=True
                )
                active_agent_thread.start()
                
            elif payload.get("action") == "stop":
                agent_stop_event.set()
                await websocket.send_json({"type": "status", "data": "stopped"})

    except WebSocketDisconnect:
        logging.info("WebSocket disconnected.")
        agent_stop_event.set()
    finally:
        if 'pump_task' in locals() and pump_task:
            pump_task.cancel()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
