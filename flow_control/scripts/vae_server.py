import asyncio
import io

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from flow_control.processors import Processor
from flow_control.utils.common import tensor_to_pil
from flow_control.utils.loaders import load_config_file
from flow_control.utils.logging import get_logger

# 配置日志
logger = get_logger(__name__)


class VaeServerConfig(BaseModel):
    processor: Processor
    host: str = "0.0.0.0"
    port: int = 8000


# 全局变量
processor = None
processing_lock = asyncio.Lock()  # 确保一次只处理一个batch

app = FastAPI(title="VAE Decode Server")


@app.get("/health")
async def health_check():
    """健康检查端点"""
    if processor is None:
        raise HTTPException(status_code=503, detail="Processor not loaded")
    return {"status": "healthy"}


@app.post("/vae_decode")
async def vae_decode(file: UploadFile):
    """
    VAE解码端点
    接收一个包含batch字典的pickle文件
    返回解码后的PNG图像
    """
    if processor is None:
        raise HTTPException(status_code=503, detail="Processor not loaded")

    # 使用锁确保一次只处理一个batch
    async with processing_lock:
        try:
            logger.info(f"Receiving batch from {file.filename}")

            # 读取上传的文件
            content = await file.read()

            # 使用torch.load反序列化batch
            batch = torch.load(io.BytesIO(content), map_location="cpu")

            # 验证batch包含必要的键
            if "noisy_latents" not in batch:
                raise HTTPException(
                    status_code=400, detail="Batch must contain 'noisy_latents' key"
                )

            logger.info(
                f"Processing batch with latents shape: {batch['noisy_latents'].shape}"
            )

            # 执行解码
            image = processor.decode_output(batch["noisy_latents"], batch)
            pil_image = tensor_to_pil(image)

            # 将PIL图像转换为PNG字节流
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)

            logger.info("Batch processed successfully")

            # 返回PNG图像
            return Response(content=img_byte_arr.read(), media_type="image/png")

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            raise HTTPException(
                status_code=500, detail=f"Error processing batch: {str(e)}"
            ) from e


def start_server(config_path: str):
    """启动服务器的入口函数"""
    # 加载配置
    config = VaeServerConfig(**load_config_file(config_path))

    # 更新全局processor
    global processor
    processor = config.processor
    processor.load_models(["decode"])

    # 启动服务器
    logger.info(f"Starting VAE server on {config.host}:{config.port}")
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python vae_server.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    start_server(config_path)


if __name__ == "__main__":
    main()
