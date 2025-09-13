
await Promise.all(
    Array(300).fill(0).map(_ =>
        fetch('http://localhost:8000/movies-recom', {
            method: 'POST',
            body: JSON.stringify({
                userId: 1,
                genres: [],
                start: 0,
                count: 10,
                model: "xgb_cpu",
                temp: 0,
            }),
            headers: {
                'Content-Type': 'application/json',
            },
        }))
).then(responses => Promise.all(responses.map(r => r.json())))
    // .then(data => console.log(data))
    .catch(error => console.error('Error:', error));