# Dockerfile for PRSFNN Julia module
FROM julia:1.10

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*
# Set workdir
WORKDIR /app

# Copy Julia project files
COPY Project.toml Manifest.toml ./
COPY src/ ./src/
COPY test/ ./test/

# (Optional) Copy any scripts or entrypoints you want to expose
# COPY scripts/ ./scripts/

# Install dependencies
RUN julia --project=. -e 'using Pkg; Pkg.instantiate()'
#RUN julia --project=. -e 'using PackageCompiler; create_sysimage(["PRSFNN"]; sysimage_path = "/app/PRSFNN.so")'
RUN julia --project=. -O3 -t 2 -e 'using PRSFNN'

# ENTRYPOINT ["julia", "-J/app/PRSFNN.so",  "--project=/app"]
ENTRYPOINT ["julia",  "--project=/app"]

# (Optional) Set default command, e.g., run a script or open Julia REPL
# CMD ["julia"]

# For example, to run main.jl by default:
# CMD ["julia", "src/main.jl"]
