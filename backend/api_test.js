

if (process.argv.length < 3)
    throw Error("Provide number of requests")
const n = Number(process.argv[2])

await Promise.all(
    Array(n).fill(0).map(_ =>
        fetch('http://localhost:8000/movies-recom', {
            method: 'POST',
            body: JSON.stringify({
                userId: 0,
                genres: ["Action"],
                start: 0,
                count: 10,
                model: "xgb_cpu",
                temp: 0.4,
            }),
            headers: {
                'Content-Type': 'application/json',
            },
        }))
).then(responses => Promise.all(responses.map(r => r.json())))
    // .then(data => console.log(data))
    .catch(error => console.error('Error:', error));
await Promise.all(
    Array(n).fill(0).map(_ =>
        fetch('http://localhost:8000/similar-movies', {
            method: 'POST',
            body: JSON.stringify({
                userId: 0,
                movieIds: [10, 182],
                start: 0,
                count: 10,
                model: "xgb_cpu",
                temp: 0.4,
            }),
            headers: {
                'Content-Type': 'application/json',
            },
        }))
).then(responses => Promise.all(responses.map(r => r.json())))
    // .then(data => console.log(data))
    .catch(error => console.error('Error:', error));