#!/usr/bin/env python3
"""
Simple test script to validate Gemini SDK integration
Usage: python test_gemini.py YOUR_API_KEY [MODEL_NAME]
"""
import sys
import os
import asyncio
from dotenv import load_dotenv

# Load env vars first
load_dotenv()

# Add backend directory to path when this script is run directly
sys.path.insert(0, os.path.dirname(__file__))

async def test_gemini():
    if len(sys.argv) < 2:
        print("❌ 錯誤：需要提供 Gemini API 金鑰")
        print("用法: python test_gemini.py YOUR_GEMINI_API_KEY [MODEL_NAME]")
        print("      例如: python test_gemini.py AIzaSy... gemini-flash")
        return False
    
    api_key = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else os.getenv("GEMINI_MODEL", "gemini-pro")
    
    os.environ["GEMINI_API_KEY"] = api_key
    os.environ["GEMINI_MODEL"] = model_name
    
    try:
        import llm
        
        # 檢查 gemini 是否在可用模型中
        if "gemini" not in llm.AVAILABLE_MODELS:
            print("❌ Gemini 未在可用模型中")
            return False
        
        print("✓ Gemini 模型已配置")
        print(f"✓ 可用模型: {list(llm.AVAILABLE_MODELS.keys())}")
        
        # 測試簡單的 chat_completion
        test_messages = [
            {"role": "user", "content": "What is 2+2?"}
        ]
        
        print("\n正在測試 Gemini API 調用...")
        response = await llm.chat_completion(
            test_messages,
            temperature=0.7,
            max_tokens=100,
            model_name="gemini"
        )
        
        print(f"✓ Gemini 回應: {response}")
        print("\n✅ Gemini 集成測試成功！")
        return True
        
    except Exception as e:
        print(f"❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_gemini())
    sys.exit(0 if success else 1)
