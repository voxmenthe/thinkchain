from tools.base import BaseTool
import requests
import json

class WeatherTool(BaseTool):
    name = "weathertool"
    description = """
    Gets current weather information for any location worldwide. Returns temperature, 
    weather conditions, humidity, wind speed and direction. 
    
    Use this tool when users ask about:
    - Current weather in any city/location
    - Temperature anywhere
    - Weather conditions (sunny, cloudy, rainy, etc.)
    - "What's the weather like in [location]?"
    """
    input_schema = {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state/country (e.g., 'Cross Lanes, WV' or 'London, UK')"
            },
            "units": {
                "type": "string",
                "description": "Temperature units: 'fahrenheit', 'celsius', or 'kelvin'",
                "enum": ["fahrenheit", "celsius", "kelvin"],
                "default": "fahrenheit"
            }
        },
        "required": ["location"]
    }

    def execute(self, **kwargs) -> str:
        location = kwargs.get("location")
        units = kwargs.get("units", "fahrenheit")
        
        # Note: This is a demo implementation. In production, you would:
        # 1. Get an API key from OpenWeatherMap (free)
        # 2. Replace this with actual API calls
        # 3. Handle API errors properly
        
        # For now, we'll use a free weather service or return demo data
        try:
            # Using wttr.in as a free alternative (no API key needed)
            # Format: plain text with temperature info
            url = f"https://wttr.in/{location}?format=j1"
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            current = data['current_condition'][0]
            
            # Extract weather data
            temp_c = int(current['temp_C'])
            temp_f = int(current['temp_F'])
            feels_like_c = int(current['FeelsLikeC'])
            feels_like_f = int(current['FeelsLikeF'])
            humidity = current['humidity']
            weather_desc = current['weatherDesc'][0]['value']
            wind_mph = current['windspeedMiles']
            wind_dir = current['winddir16Point']
            
            # Format temperature based on requested units
            if units.lower() == "celsius":
                temp = f"{temp_c}°C"
                feels_like = f"{feels_like_c}°C"
            elif units.lower() == "kelvin":
                temp_k = temp_c + 273.15
                feels_like_k = feels_like_c + 273.15
                temp = f"{temp_k:.1f}K"
                feels_like = f"{feels_like_k:.1f}K"
            else:  # fahrenheit (default)
                temp = f"{temp_f}°F"
                feels_like = f"{feels_like_f}°F"
            
            result = f"""🌤️ Weather for {location}:
Temperature: {temp} (feels like {feels_like})
Conditions: {weather_desc}
Humidity: {humidity}%
Wind: {wind_mph} mph {wind_dir}"""
            
            return result
            
        except requests.RequestException as e:
            return f"❌ Error fetching weather data: {str(e)}"
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            return f"❌ Error parsing weather data: {str(e)}"
        except Exception as e:
            return f"❌ Unexpected error: {str(e)}"