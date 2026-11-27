import argparse
import json
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv
from openai import OpenAI

# Load the OpenAI API key from the .env file and create the OpenAI client.
load_dotenv()
client = OpenAI()

# Get the current time in a specified location and timezone.
def get_current_time(location: str, timezone: str) -> dict[str, str]:
    try:
        target_timezone = ZoneInfo(timezone)
    except ZoneInfoNotFoundError as error:
        raise ValueError(f"Unknown IANA timezone: {timezone}") from error

    system_time = datetime.now().astimezone()
    location_time = system_time.astimezone(target_timezone)
    return {
        "location": location,
        "system_time": system_time.isoformat(timespec="seconds"),
        "location_time": location_time.isoformat(timespec="seconds"),
        "timezone": timezone,
    }

# Define the tool for getting the current time.
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": (
                "Get the current system time and the current time in a requested "
                "city or country. Choose the appropriate IANA timezone for the "
                "location, for example Europe/Prague."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The requested city and/or country.",
                    },
                    "timezone": {
                        "type": "string",
                        "description": "The location's IANA timezone identifier.", #IANA = Internet Assigned Numbers Authority
                    },
                },
                "required": ["location", "timezone"],
                "additionalProperties": False,
            },
        },
    }
]

# Function to ask the model for the current time in a specified location.
def ask_for_time(question: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful time assistant. For a request about the current "
                "time, call get_current_time exactly once. Use the user's city or "
                "country to choose its IANA timezone. Do not use external services. "
                "After receiving the tool result, clearly report both the system "
                "time and the requested location's time."
            ),
        },
        {"role": "user", "content": question},
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice="required",
        parallel_tool_calls=False,
    )
    assistant_message = response.choices[0].message

    if not assistant_message.tool_calls:
        raise RuntimeError("The model did not call the time tool.")

    tool_call = assistant_message.tool_calls[0]
    if tool_call.function.name != "get_current_time":
        raise RuntimeError(f"Unexpected tool call: {tool_call.function.name}")

    arguments = json.loads(tool_call.function.arguments)
    result = get_current_time(
        location=arguments["location"],
        timezone=arguments["timezone"],
    )

# Keep a record of the tool call and its result so the AI can understand it.
    messages.extend(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": json.dumps(result),
            },
        ]
    )

    final_response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )
    return final_response.choices[0].message.content or ""

# Read the question typed after the script name and print the AI's answer.
parser = argparse.ArgumentParser(
    description="Get the current system time and time in another location."
)
parser.add_argument(
    "question",
    help='For example: "What time is it in Tokyo, Japan?"',
)
args = parser.parse_args()

print(ask_for_time(args.question))

# To test it out, run in terminal: python3 02_current_time.py "What time is it in Tokyo, Japan?"