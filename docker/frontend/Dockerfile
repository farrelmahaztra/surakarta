FROM node:20-alpine

WORKDIR /app

COPY package.json package-lock.json /app/

COPY . /app/

RUN yarn build