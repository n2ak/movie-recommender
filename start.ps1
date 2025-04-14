cd frontend
# rm vlms -r -Force
#npx prisma migrate dev --name init # migrate changes to db
#python -m prisma generate # generate python files
pnpm run db:generate
pnpm run db:migrate

cd ../back
python scripts/filldb.py
cd ..