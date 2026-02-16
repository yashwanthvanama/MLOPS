#!/bin/bash

set -e

export APP_DIR=/opt/flower-classifier-app
mkdir -p $APP_DIR
cd $APP_DIR

apt update -y
apt install -y git python3 python3-pip python3-venv nginx

git clone https://github.com/yashwanthvanama/MLOPS.git .

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip

python3 -m pip install -r requirements.txt

python3 flower_classifier.py

cat >/etc/systemd/system/flower-classifier.service <<'EOF'
[Unit]
Description=Flower Classifier Service
After=network.target

[Service]
User=ubuntu
Group=ubuntu
WorkingDirectory=/opt/flower-classifier-app
ExecStart=/opt/flower-classifier-app/.venv/bin/gunicorn --workers 3 --bind 127.0.0.1:6000 app:app
Restart=always

[Install]
WantedBy=multi-user.target
EOF

cat >/etc/nginx/conf.d/flower-classifier.conf <<'EOF'
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:6000/predict;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_connect_timeout 60s;
        proxy_read_timeout 120s;
    }
}
EOF

if [ -f /etc/nginx/sites-enabled/default ] || [ -L /etc/nginx/sites-enabled/default ]; then
    rm -f /etc/nginx/sites-enabled/default || true
fi

systemctl daemon-reload
systemctl enable flower-classifier
systemctl start flower-classifier
systemctl enable nginx
systemctl restart nginx