# --- deps ---
FROM node:22-slim AS deps
WORKDIR /srv
COPY package.json package-lock.json ./
RUN npm ci

# --- build ---
FROM node:22-slim AS builder
WORKDIR /srv
COPY --from=deps /srv/node_modules ./node_modules
COPY . .
RUN npm run build

# --- run ---
FROM node:22-slim AS runner
WORKDIR /srv
ENV NODE_ENV=production
ENV PORT=3000
ENV HOSTNAME=0.0.0.0

COPY --from=builder /srv/public ./public
COPY --from=builder /srv/.next/standalone ./
COPY --from=builder /srv/.next/static ./.next/static

EXPOSE 3000
CMD ["node", "server.js"]
