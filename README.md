# streamlit-web-ml
利用streamlit快速搭建机器学习web展示demo

# 部署地址
https://streamlit-web-ml.herokuapp.com/

## 怎样在Heroku发布Streamlit Apps

* 1. Create An Account Heroku by signing up.
* 2. Install Heroku CLI
* 3. Create Your Github Repository for your app
* 4. Build your app
* 5. Login to Heroku From the CLI

```sh 
heroku Login
```

* 6. Create Your 3 Required Files(setup.sh,Procfile,requirements.txt)
  * (1) Code for setup.sh
```sh
mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
```

  * (2) Code for setup.sh (Alternate with no credentials.toml)
```sh
mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

  * (3) Code For Procfile
```sh
web: sh setup.sh && streamlit run your_app.py
```

* 7. Create App with CLI
```sh
heroku create name-of-your-app
```

* 8. Commit and Push Your Code to Github
```sh
git add your app 
git commit -m "your comment description"
git push
```

* 9. Push To Heroku to Deploy
```sh
git push heroku master
```


# 所需要的配置文件
* Procfile
* setup.sh
* requirements.txt
