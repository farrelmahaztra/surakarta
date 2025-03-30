# Surakarta

An implementation of the traditional Indonesian board game with AI opponents.

## Features

- Surakarta game engine
- Multiple AI opponents with different strategies:
  - Minimax with alpha-beta pruning
  - DQN
  - Greedy and defensive strategies
- Web interface for online play

## Getting Started

### Using Docker

```bash
# Copy environment template
cp env.template .env

# Start all services
docker-compose up -d
```

### Setup

```bash
# Start frontend
cd web/frontend
npm run dev

# Create a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate
uv pip install .

# Migrate DB
python manage.py migrate

# Start backend
python manage.py runserver

# Train your own AI (optional)
python train.py --opponent minimax --episodes 1000
```

### Requirements

- Python 3.11
- Python dependencies are in `pyproject.toml`
- React dependencies are in `package.json`

## License

See [LICENSE](LICENSE) file for details.