from pydantic import BaseSettings

class Settings(BaseSettings):
    user_name: str
    password: str
    name_database: str
    address: str
    port: int
       
    class Config:
        env_file = '.env'
        