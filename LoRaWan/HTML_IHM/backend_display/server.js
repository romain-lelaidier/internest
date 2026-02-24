/*
Js server to track evolution of the csv file being written and send them to the html frontend.
*/

let csv_path = "/mnt/c/Users/colin/COLIN/Mines/TI_IDS/Internest/LoRaWan/python_serial/monitoring_data.csv"

const express = require('express');
const fs = require('fs');
const app = express();
const port = 3000;

// Allow CORS (if frontend is on a different port/domain)
app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
    next();
});

// Track the last read position
let lastPosition = 0;

// Endpoint to get new lines
app.get('/new-lines', (req, res) => {
    fs.stat(csv_path, (err, stats) => {
        if (err) {
            console.error("Error reading file:", err);
            return res.status(500).send("Error reading file");
        }

        const fileSize = stats.size;
        if (fileSize <= lastPosition) {
            return res.send(""); // No new lines
        }

        const stream = fs.createReadStream(csv_path, {
            start: lastPosition,
            end: fileSize
        });

        let newLines = "";
        stream.on('data', (chunk) => {
            newLines += chunk.toString();
        });

        stream.on('end', () => {
            lastPosition = fileSize;
            res.send(newLines);
        });

        stream.on('error', (err) => {
            console.error("Stream error:", err);
            res.status(500).send("Stream error");
        });
    });
});

// Start the server on 0.0.0.0 (accessible from Windows)
app.listen(port, '0.0.0.0', () => {
    console.log(`Backend running at http://0.0.0.0:${port}`);
});
