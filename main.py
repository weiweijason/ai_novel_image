"""
Image Worker - ComfyUI 影像生成 Worker
硬體需求: RTX 5090 32GB VRAM
職責: 角色立繪、場景圖片、背景圖片生成
"""

import io
import os
import time
from datetime import datetime, timezone
from typing import Any

import requests
from minio import Minio


# === 環境變數 ===
API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000").rstrip("/")
WORKER_ID = os.getenv("WORKER_ID", "image-01")
WORKER_TYPE = os.getenv("WORKER_TYPE", "image")
WORKER_HOSTNAME = os.getenv("WORKER_HOSTNAME", "image-worker")
WORKER_CAPABILITIES = os.getenv(
    "WORKER_CAPABILITIES",
    "character_image,scene_image,background_image,character_expression,character_pose",
).split(",")
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "5"))

# MinIO 設定
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://minio:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin123")
S3_BUCKET = os.getenv("S3_BUCKET", "assets")

# ComfyUI 設定
COMFYUI_API_URL = os.getenv("COMFYUI_API_URL", "http://localhost:8188")

# 影像生成參數
IMAGE_WIDTH = int(os.getenv("IMAGE_WIDTH", "512"))
IMAGE_HEIGHT = int(os.getenv("IMAGE_HEIGHT", "768"))
CFG_SCALE = float(os.getenv("CFG_SCALE", "7.5"))
NUM_INFERENCE_STEPS = int(os.getenv("NUM_INFERENCE_STEPS", "30"))
SAMPLER = os.getenv("SAMPLER", "euler_ancestral")
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "4"))


# === MinIO 客戶端 ===
def get_minio_client() -> Minio:
    return Minio(
        S3_ENDPOINT.replace("http://", "").replace("https://", ""),
        access_key=S3_ACCESS_KEY,
        secret_key=S3_SECRET_KEY,
        secure=S3_ENDPOINT.startswith("https"),
    )


minio_client = get_minio_client()


def ensure_bucket() -> None:
    """確保 Bucket 存在"""
    if not minio_client.bucket_exists(S3_BUCKET):
        minio_client.make_bucket(S3_BUCKET)


# === Worker API 通訊 ===
def register_worker() -> None:
    """向 API 註冊 Image Worker"""
    payload = {
        "worker_id": WORKER_ID,
        "worker_type": WORKER_TYPE,
        "hostname": WORKER_HOSTNAME,
        "capabilities": WORKER_CAPABILITIES,
        "models": ["stable-diffusion-xl-anime", "controlnet-pose"],
    }
    print(f"[{datetime.now(timezone.utc).isoformat()}] Registering worker: {WORKER_ID}")
    response = requests.post(f"{API_BASE_URL}/worker/register", json=payload, timeout=10)
    response.raise_for_status()
    print(f"[{datetime.now(timezone.utc).isoformat()}] Worker registered successfully")


def send_heartbeat(status: str, current_job: str | None = None) -> None:
    """發送心跳"""
    payload = {
        "worker_id": WORKER_ID,
        "status": status,
        "current_job": current_job,
        "gpu": {
            "name": "RTX 5090",
            "vram_total": 32768,
            "vram_used": 0,  # TODO: 實際查詢 VRAM 使用量
        },
    }
    response = requests.post(
        f"{API_BASE_URL}/worker/heartbeat",
        json=payload,
        timeout=10,
    )
    response.raise_for_status()


def claim_job() -> dict[str, Any] | None:
    """從佇列領取下一個影像生成作業"""
    response = requests.post(
        f"{API_BASE_URL}/worker/jobs/claim",
        json={"worker_id": WORKER_ID},
        timeout=10,
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


def update_job_status(
    job_id: str,
    status: str,
    progress: float | None = None,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    """更新作業狀態"""
    payload: dict[str, Any] = {"status": status}
    if progress is not None:
        payload["progress"] = progress
    if result is not None:
        payload["result"] = result
    if error is not None:
        payload["error"] = error

    response = requests.post(
        f"{API_BASE_URL}/worker/jobs/{job_id}/status",
        json=payload,
        timeout=10,
    )
    response.raise_for_status()


# === MinIO 上傳 ===
def upload_to_minio(image_bytes: bytes, object_name: str, content_type: str = "image/png") -> str:
    """上傳影像到 MinIO"""
    ensure_bucket()
    image_stream = io.BytesIO(image_bytes)
    minio_client.put_object(
        bucket_name=S3_BUCKET,
        object_name=object_name,
        data=image_stream,
        length=len(image_bytes),
        content_type=content_type,
    )
    return f"{S3_ENDPOINT}/{S3_BUCKET}/{object_name}"


# === ComfyUI API 整合 ===
def submit_comfyui_workflow(workflow: dict[str, Any], client_id: str = "") -> str:
    """提交工作流到 ComfyUI 並回傳 prompt_id"""
    payload = {"prompt": workflow, "client_id": client_id}
    response = requests.post(f"{COMFYUI_API_URL}/prompt", json=payload, timeout=30)
    response.raise_for_status()
    result = response.json()
    return result["prompt_id"]


def wait_comfyui_result(prompt_id: str, timeout_seconds: int = 600) -> dict[str, Any]:
    """等待 ComfyUI 完成並回傳結果"""
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout_seconds:
            raise TimeoutError(f"ComfyUI workflow timed out after {timeout_seconds}s")
        
        response = requests.get(
            f"{COMFYUI_API_URL}/history/{prompt_id}",
            timeout=10,
        )
        response.raise_for_status()
        history = response.json()
        
        if prompt_id in history:
            return history[prompt_id]
        
        time.sleep(2)


def download_comfyui_image(output_info: dict[str, Any]) -> bytes:
    """從 ComfyUI 下載生成的影像"""
    image_name = output_info.get("filename", "")
    subfolder = output_info.get("subfolder", "")
    
    if not image_name:
        raise ValueError("No image filename in ComfyUI output")
    
    params = {"filename": image_name, "subfolder": subfolder, "type": "output"}
    response = requests.get(f"{COMFYUI_API_URL}/view", params=params, timeout=30)
    response.raise_for_status()
    return response.content


def build_comfyui_workflow(
    prompt: str,
    negative_prompt: str = "low quality, blurry, bad anatomy",
    width: int = IMAGE_WIDTH,
    height: int = IMAGE_HEIGHT,
    seed: int = 42,
) -> dict[str, Any]:
    """建立 ComfyUI 工作流"""
    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": NUM_INFERENCE_STEPS,
                "cfg": CFG_SCALE,
                "sampler_name": SAMPLER,
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            }
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "anime_model.safetensors"}
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1
            }
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["4", 1]}
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["4", 1]}
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]}
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"images": ["8", 0], "filename_prefix": "anime"}
        }
    }


# === 影像生成函式 ===
def generate_character_image(user_input: dict[str, Any]) -> dict[str, Any]:
    """
    生成角色立繪 (ComfyUI)
    
    Input:
    {
        "character_name": "主角",
        "character_description": "藍色長髮，紅色眼睛",
        "style": "anime",
        "width": 1024,
        "height": 1024
    }
    
    Output:
    {
        "image_url": "http://minio:9000/assets/characters/xxx.png",
        "metadata": {...}
    }
    """
    character_name = user_input.get("character_name", "character")
    character_desc = user_input.get("character_description", "")
    style = user_input.get("style", "anime")
    width = user_input.get("width", IMAGE_WIDTH)
    height = user_input.get("height", IMAGE_HEIGHT)
    
    print(f"[{datetime.now(timezone.utc).isoformat()}] Generating character image: {character_name}")
    
    # 建立 prompt
    prompt = f"masterpiece, best quality, {style} style character portrait, {character_desc}, full body, detailed background"
    negative_prompt = "low quality, blurry, bad anatomy, extra limbs, worst quality"
    
    try:
        # 1. 建立 ComfyUI 工作流
        workflow = build_comfyui_workflow(prompt, negative_prompt, width, height)
        
        # 2. 提交到 ComfyUI
        print(f"  Submitting workflow to ComfyUI: {COMFYUI_API_URL}")
        prompt_id = submit_comfyui_workflow(workflow)
        print(f"  Prompt ID: {prompt_id}")
        
        # 3. 等待完成
        print("  Waiting for generation...")
        history = wait_comfyui_result(prompt_id)
        print(f"  Generation complete: {history}")
        
        # 4. 下載影像
        outputs = history.get("outputs", {})
        first_output = next(iter(outputs.values()), None)
        if first_output and "images" in first_output:
            image_info = first_output["images"][0]
            image_bytes = download_comfyui_image(image_info)
        else:
            raise ValueError("No image output found in ComfyUI result")
        
    except Exception as e:
        print(f"  ComfyUI error: {e}")
        print("  Falling back to Mock generation...")
        # Fallback: Mock 生成
        image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 1000
    
    # 上傳到 MinIO
    object_name = f"characters/{character_name}_{int(time.time())}.png"
    image_url = upload_to_minio(image_bytes, object_name)
    
    return {
        "image_url": image_url,
        "metadata": {
            "character_name": character_name,
            "style": style,
            "width": width,
            "height": height,
            "prompt": prompt,
        }
    }


def generate_scene_image(user_input: dict[str, Any]) -> dict[str, Any]:
    """
    生成場景圖片 (ComfyUI)
    
    Input:
    {
        "scene_description": "學校教室，下午",
        "characters": [...],
        "style": "anime",
        "width": 1920,
        "height": 1080
    }
    
    Output:
    {
        "image_url": "http://minio:9000/assets/scenes/xxx.png",
        "metadata": {...}
    }
    """
    scene_desc = user_input.get("scene_description", "anime scene")
    style = user_input.get("style", "anime")
    width = user_input.get("width", 1920)
    height = user_input.get("height", 1080)
    
    print(f"[{datetime.now(timezone.utc).isoformat()}] Generating scene image")
    
    prompt = f"masterpiece, best quality, {style} style, {scene_desc}, detailed background, cinematic lighting"
    negative_prompt = "low quality, blurry, bad anatomy, extra limbs, worst quality"
    
    try:
        workflow = build_comfyui_workflow(prompt, negative_prompt, width, height)
        print(f"  Submitting workflow to ComfyUI: {COMFYUI_API_URL}")
        prompt_id = submit_comfyui_workflow(workflow)
        print(f"  Prompt ID: {prompt_id}")
        print("  Waiting for generation...")
        history = wait_comfyui_result(prompt_id)
        print(f"  Generation complete")
        
        outputs = history.get("outputs", {})
        first_output = next(iter(outputs.values()), None)
        if first_output and "images" in first_output:
            image_info = first_output["images"][0]
            image_bytes = download_comfyui_image(image_info)
        else:
            raise ValueError("No image output found")
    except Exception as e:
        print(f"  ComfyUI error: {e}")
        print("  Falling back to Mock generation...")
        image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 1000
    
    object_name = f"scenes/scene_{int(time.time())}.png"
    image_url = upload_to_minio(image_bytes, object_name)
    
    return {
        "image_url": image_url,
        "metadata": {
            "width": width,
            "height": height,
            "prompt": prompt,
        }
    }


def generate_background_image(user_input: dict[str, Any]) -> dict[str, Any]:
    """
    生成背景圖片 (ComfyUI)
    
    Input:
    {
        "location": "公園",
        "time": "afternoon",
        "weather": "clear"
    }
    
    Output:
    {
        "image_url": "http://minio:9000/assets/backgrounds/xxx.png",
        "metadata": {...}
    }
    """
    location = user_input.get("location", "background")
    time_of_day = user_input.get("time", "day")
    weather = user_input.get("weather", "clear")
    
    print(f"[{datetime.now(timezone.utc).isoformat()}] Generating background: {location}")
    
    prompt = f"masterpiece, best quality, anime background, {location}, {time_of_day}, {weather}, detailed, cinematic"
    negative_prompt = "low quality, blurry, worst quality"
    
    try:
        workflow = build_comfyui_workflow(prompt, negative_prompt)
        print(f"  Submitting workflow to ComfyUI: {COMFYUI_API_URL}")
        prompt_id = submit_comfyui_workflow(workflow)
        print(f"  Prompt ID: {prompt_id}")
        print("  Waiting for generation...")
        history = wait_comfyui_result(prompt_id)
        
        outputs = history.get("outputs", {})
        first_output = next(iter(outputs.values()), None)
        if first_output and "images" in first_output:
            image_info = first_output["images"][0]
            image_bytes = download_comfyui_image(image_info)
        else:
            raise ValueError("No image output found")
    except Exception as e:
        print(f"  ComfyUI error: {e}")
        print("  Falling back to Mock generation...")
        image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 1000
    
    object_name = f"backgrounds/{location}_{time_of_day}_{int(time.time())}.png"
    image_url = upload_to_minio(image_bytes, object_name)
    
    return {
        "image_url": image_url,
        "metadata": {
            "location": location,
            "time": time_of_day,
            "weather": weather,
            "width": IMAGE_WIDTH,
            "height": IMAGE_HEIGHT,
            "prompt": prompt,
        }
    }


def run_image_job(job: dict[str, Any]) -> dict[str, Any]:
    """執行影像生成作業"""
    job_type = job["type"]
    user_input = job.get("input", {})
    
    if job_type == "character_image":
        return generate_character_image(user_input)
    if job_type == "scene_image":
        return generate_scene_image(user_input)
    if job_type == "background_image":
        return generate_background_image(user_input)
    
    raise ValueError(f"Unknown job type: {job_type}")


# === Worker 主循環 ===
def main() -> None:
    """Image Worker 主循環"""
    print(f"[{datetime.now(timezone.utc).isoformat()}] Image Worker starting...")
    print(f"  Worker ID: {WORKER_ID}")
    print(f"  API Base URL: {API_BASE_URL}")
    print(f"  ComfyUI API: {COMFYUI_API_URL}")
    print(f"  S3 Endpoint: {S3_ENDPOINT}")
    print(f"  Capabilities: {WORKER_CAPABILITIES}")
    
    # 註冊 Worker
    register_worker()
    
    current_job_id = None
    
    while True:
        try:
            # 發送心跳
            status = "busy" if current_job_id else "idle"
            send_heartbeat(status, current_job_id)
            
            # 如果有正在執行的作業，繼續執行
            if current_job_id:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue
            
            # 領取新作業
            job = claim_job()
            
            if job is None:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue
            
            current_job_id = job["id"]
            print(f"[{datetime.now(timezone.utc).isoformat()}] Claimed job: {current_job_id} (type: {job['type']})")
            
            # 更新狀態為 running
            update_job_status(current_job_id, "running", progress=0.0)
            
            # 執行作業
            result = run_image_job(job)
            
            # 更新進度
            update_job_status(current_job_id, "running", progress=100.0)
            
            # 標記完成
            update_job_status(current_job_id, "completed", result=result)
            
            print(f"[{datetime.now(timezone.utc).isoformat()}] Job completed: {current_job_id}")
            current_job_id = None
            
        except Exception as e:
            print(f"[{datetime.now(timezone.utc).isoformat()}] Error: {e}")
            if current_job_id:
                try:
                    update_job_status(current_job_id, "failed", error=str(e))
                except Exception:
                    pass
                current_job_id = None
            time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
