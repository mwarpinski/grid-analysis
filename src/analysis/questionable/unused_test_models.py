import asyncio
import os
import sys
from pathlib import Path

# Load .env
env_path = Path("/home/matt/quantshop/smartgrid/grid-analysis/.env")
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

import anthropic

async def test_model(model_name):
    client = anthropic.AsyncAnthropic()
    print(f"Testing model: {model_name}...")
    try:
        response = await client.messages.create(
            model=model_name,
            max_tokens=10,
            messages=[{"role": "user", "content": "Hello"}]
        )
        print(f"  SUCCESS! Response: {response.content[0].text.strip()}")
        return True
    except Exception as e:
        print(f"  FAILED with: {e}")
        return False

async def main():
    models = [
        "claude-3-haiku-20240307",
        "claude-3-5-haiku-20241022",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-latest",
        "claude-3-7-sonnet-latest",
    ]
    for model in models:
        await test_model(model)
        print("-" * 40)

if __name__ == "__main__":
    asyncio.run(main())
