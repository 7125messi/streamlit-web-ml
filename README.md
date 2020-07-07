# streamlit-web-ml
利用streamlit快速搭建机器学习web展示demo

# 部署地址
https://streamlit-web-ml.herokuapp.com/

## 怎样在Heroku发布Streamlit Apps

* Create An Account Heroku by signing up.
* Install Heroku CLI
* Create Your Github Repository for your app
* Build your app
* Login to Heroku From the CLI

```sh 
heroku Login
```

* Create Your 3 Required Files(`setup.sh`,`Procfile`,`requirements.txt`)
> Code for `setup.sh`
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

> Code for `setup.sh` (Alternate with no credentials.toml)
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

> Code For `Procfile`
```sh
web: sh setup.sh && streamlit run your_app.py
```

* Create App with CLI
```sh
heroku create name-of-your-app
```

* Commit and Push Your Code to Github
```sh
git add your app 
git commit -m "your comment description"
git push
```

* Push To Heroku to Deploy
```sh
git push heroku master
```


# 所需要的配置文件
* Procfile
* setup.sh
* requirements.txt
