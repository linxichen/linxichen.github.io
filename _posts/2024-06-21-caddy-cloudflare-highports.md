---
title: "Setting Up Caddy for Reverse Proxy with Cloudflare on Non-Standard Port"
date: 2024-06-21T01:20:00-04:00
categories:
  - Blog
tags:
  - Cloudflare
  - Caddy
  - homelab
---

In this tutorial, we will walk through the steps to set up Caddy as a reverse proxy server and configure Cloudflare to communicate with your origin server using a non-standard port (8443 instead of 443). This is achieved using a feature called **Origin Rules** by Cloudflare.


### Prerequisites

1. A domain managed by Cloudflare.
2. A server with Caddy installed.
3. Basic knowledge of how to use Cloudflare and Caddy.

### Step 1: Install Caddy

If you haven't already installed Caddy, you can do so by following the instructions [here](https://caddyserver.com/docs/install).
I use a docker compose installation which has cloudflarends plugin installed (third-party docker image, use at your own risk). 

```yaml
services:
  caddy:
    # guide by this:
    # https://caddy.community/t/how-to-guide-caddy-v2-cloudflare-dns-01-via-docker/8007
    image: slothcroissant/caddy-cloudflaredns:latest
    restart: unless-stopped
    environment:
      - CLOUDFLARE_EMAIL=my@example.com
      - CLOUDFLARE_API_TOKEN=tokentoken
      - ACME_AGREE=true
    ports:
      - "80:80"
      - "443:443"
      - "443:443/udp"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile
      - caddy_data:/data
      - caddy_config:/config

volumes:
  caddy_data:
  caddy_config:
```


Caddy is configured to listen to the usual 443 and 80 ports on a VM. However, when CloudFlare try to reach us from WAN we don't want it to use 443 because it's occupied.

### Step 2: Configure Cloudflare Origin Rules

We want to make sure

1. **Log in to Cloudflare**: Go to your Cloudflare dashboard and select your domain.
2. **Navigate to Origin Rules**: Go to the 'Rules' section and select 'Origin Rules'.
3. **Create a New Rule**: Notice how I route all traffic to 8443 port of origin server ![See this pic](/assets/images/origin-rules.png)

### Step 3: Update DNS Settings

In your Cloudflare dashboard, ensure your DNS settings for your domain are set up correctly. Your `A` or `CNAME` records should be proxied (orange cloud icon).

### Step 4: Configure Port Forwarding

So traffic from you wan IP at port 8443 is routed to Caddy server's 443 port.

### Step 5: Configure Caddy

Create or edit your Caddyfile to configure the reverse proxy. Below is an example of how to set up Caddy to proxy requests to your backend server.

```
sub.mydomain.com:443 {
        reverse_proxy http://10.24.99.88:8443
        tls name@example.com {
                dns cloudflare {env.CLOUDFLARE_API_TOKEN}
                resolvers 1.1.1.1
        }
}

```

### Step 6: Test Your Setup

After configuring both Caddy and Cloudflare, it's time to test your setup.

1. **Start Service**: Restart Caddy to apply the changes.
    ```bash
    docker compose up
    ```
2. **Restart**: Restart the stack after any changes in Caddyfile.
    ```bash
    docker compose restart
    ```

The usual suggestion is to run `caddy reload` inside the docker container but I couldn't get it to grab new SSL certificate this way. Restart seems just easier.

3. **Check Connectivity**: Ensure that your site is accessible through HTTPS and that Cloudflare is communicating with your origin server on port 8443.

### Conclusion

By following these steps, you should have successfully set up Caddy as a reverse proxy with Cloudflare using a non-standard port for origin communication. This setup enhances security and allows for flexible configuration of your web infrastructure.

If you encounter any issues or have further questions, feel free to leave a comment below.

Happy tinkering!

---


