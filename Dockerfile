# Dockerfile for PRSFNN Julia module
FROM julia:1.10

# Set workdir
WORKDIR /app

# Copy Julia project files
COPY Project.toml Manifest.toml ./
COPY src/ ./src/

# (Optional) Copy any scripts or entrypoints you want to expose
# COPY scripts/ ./scripts/

# Install dependencies
RUN julia --project=. -e 'using Pkg; Pkg.instantiate()'

ENTRYPOINT ["julia", "--project=."]

# (Optional) Set default command, e.g., run a script or open Julia REPL
# CMD ["julia"]

# For example, to run main.jl by default:
# CMD ["julia", "src/main.jl"]
