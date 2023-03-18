import datetime
from typing import Union
from pydantic import BaseModel
from pyparsing import Optional
import uvicorn

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from .application.components import predict, read_imagefile
from .application.security import Oauth2ClientCredentials, OAuth2ClientCredentialsRequestForm 
from fastapi.security import OAuth2PasswordRequestForm, HTTPBasic
from starlette.status import HTTP_401_UNAUTHORIZED
from passlib.context import CryptContext
from jose import JWTError, jwt
import secrets
import logging
from fastapi.logger import logger as fastapi_logger
from .application.security_var import ACCESS_TOKEN_EXPIRE_MINUTES, ALGORITHM, SECRET_KEY, access_control_list

app = FastAPI()



#healthcheck endpoint
@app.get("/healthcheck")
async def read_root():
    return {"status": "ok"}

#region security boiler plate



class Token(BaseModel):
    access_token: str
    token_type: str

#user must capture username and clientID
class User(BaseModel):
    username: Union[str, None] = None
    client_id: str
    hashed_secret: Union[str, None] = None
    enabled: Union[bool, None] = None


class TokenData(BaseModel):
    username: Union[str, None] = None
    client_id: str

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = Oauth2ClientCredentials(tokenUrl="token")





@app.post("/token")
async def login(request: Request,
                form_data: OAuth2ClientCredentialsRequestForm = Depends()
):
    failed_auth = HTTPException(
        status_code=400, detail="Incorrect username or password"
    )

    basic_credentials = await HTTPBasic(auto_error=False)(request)

    if form_data.client_id and form_data.client_secret:
        client_id = form_data.client_id
        client_secret = form_data.client_secret
        username = form_data.username
    elif basic_credentials:
        client_id = basic_credentials.username
        client_secret = basic_credentials.password
        username = "admin"
    else:
        raise failed_auth

    if username is None:
        username = "admin"

    api_client = authenticate_user(access_control_list, client_id, client_secret, username)
    #api_client = access_control_list.get(client_id)

    if not api_client or not api_client.enabled:
        raise failed_auth
    

    
    return {
        "access_token": create_access_token({"username": username, "client_id":client_id}, datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)),
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES*60,
    }

def create_access_token(data: dict, expires_delta: Union[datetime.timedelta, None] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.datetime.utcnow() + expires_delta
    else:
        expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_client(db, clientid: str):
    if clientid in db:
        user_dict = db[clientid].copy()
        user_dict["client_id"] = clientid  
        return User(**user_dict)



def fake_decode_token(token):
    # This doesn't provide any security at all
    # Check the next version
    user = get_client(access_control_list, token)
    return token

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code= HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("username")
        client_id: str = payload.get("client_id")
        if username is None:
            raise credentials_exception
        token_data = TokenData(client_id=client_id,username=username)
    except JWTError:
        raise credentials_exception
    user = get_client(access_control_list, clientid=token_data.client_id)
    user.username = token_data.username
    if user is None:
        raise credentials_exception
    return user

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def authenticate_user(client_db, id: str, secret: str, username: str):
    user : User = get_client(client_db, id)
    if not user:
        return False
    if not verify_password(secret, user.hashed_secret):
        return False
    user.username = username
    return user





async def get_current_active_user(current_user: User = Depends(get_current_user)):
    #if current_user.disabled:
        #raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

#endregion

@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...), current_user: User = Depends(get_current_active_user)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    prediction = predict(image)    
    return prediction


if __name__ == "__main__":
    uvicorn.run(app)