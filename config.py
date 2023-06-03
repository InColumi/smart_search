from pydantic import BaseSettings


class Settings(BaseSettings):
    user_name: str
    password: str
    name_database: str
    address: str
    port: int

    path_to_texts: str 
    class Config:
        env_file = '.env'


settings = Settings()
