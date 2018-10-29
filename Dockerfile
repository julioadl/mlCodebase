FROM library/ubuntu:18.10
FROM python:3.6

COPY /. /.

RUN pip install -r requirements.txt && \
    apt-get update && \
    apt-get install emacs24 -y && \
    apt-get install lsof -y && \
    #  apt-get install -y apache2 && \
    #  apt-get install -y apache2.2-common && \
    #  apt-get install -y apache2-mpm-prefork && \
    #  apt-get install -y apache2-utils && \
    #  apt-get install -y libexpat1 && \
    #  apt-get install -y ssl-cert && \
    #  apt-get install -y libapache2-mod-wsgi && \
    #  a2ensite 0.0.0.0.conf && \
    #  a2enmod ssl && \
    #  /etc/init.d/apache2 restart && \
    rm -rf /var/lib/apt/lists/*

EXPOSE 80:8000
