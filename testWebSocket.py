import websockets
import asyncio

async def basic_websocket():
    async with websockets.connect('ws://localhost:3000/') as connection:
        await connection.send(b'Hello, world!')
        response = await connection.recv()
        print(f"Received: {response}")

asyncio.run(basic_websocket())