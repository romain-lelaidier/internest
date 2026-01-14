import express from 'express';

const app = express();

app.get('/', async (req, res) => {
  res.send('Hello, world!');
})

const PORT = process.env.PORT_API || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});