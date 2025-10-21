# syntax=docker.io/docker/dockerfile:1
FROM node:20-alpine AS base

FROM base AS prune
RUN apk update
RUN apk add --no-cache libc6-compat
WORKDIR /app
########### install pnpm
RUN corepack enable && corepack prepare pnpm@latest --activate
ENV PNPM_HOME=/usr/local/bin
ENV PATH=$PNPM_HOME:$PATH
###########
RUN pnpm add -g turbo
COPY . .
RUN turbo prune web --docker


FROM base AS installer
RUN apk update
RUN apk add --no-cache libc6-compat
WORKDIR /app
########### install pnpm
RUN corepack enable && corepack prepare pnpm@latest --activate
ENV PNPM_HOME=/usr/local/bin
ENV PATH=$PNPM_HOME:$PATH
###########
COPY --from=prune /app/out/json/ .
RUN pnpm i --frozen-lockfile

COPY --from=prune /app/out/full/ .
RUN pnpm run db:generate
RUN pnpm dlx turbo run build 



FROM base AS runner
WORKDIR /app
########### install pnpm
RUN corepack enable && corepack prepare pnpm@latest --activate
ENV PNPM_HOME=/usr/local/bin
ENV PATH=$PNPM_HOME:$PATH
###########
ENV NODE_ENV=production
# ENV NEXT_TELEMETRY_DISABLED=1
RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs
USER nextjs
COPY --from=installer --chown=nextjs:nodejs /app/apps/web/.next/standalone ./
COPY --from=installer --chown=nextjs:nodejs /app/apps/web/.next/static ./apps/web/.next/static
COPY --from=installer --chown=nextjs:nodejs /app/apps/web/public ./apps/web/public

EXPOSE 3000
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

CMD node apps/web/server.js
