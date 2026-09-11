FROM alpine:3.22
RUN apk add --no-cache iproute2 iptables
COPY network.sh /network.sh
ENTRYPOINT ["sh", "/network.sh"]
