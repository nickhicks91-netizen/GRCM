# GRCM Contribution Policy

**CONFIDENTIAL & PROPRIETARY**

---

## Notice

GRCM (Grounded Resonant Consciousness Module) is **proprietary software** protected as a trade secret. This is NOT an open-source project.

---

## Contribution Policy

**Public contributions are NOT accepted.**

This codebase is:
- Protected as confidential trade secret
- Available for exclusive acquisition only
- Not licensed for public use, modification, or distribution
- Subject to strict confidentiality requirements

---

## Authorized Development Only

Development on this codebase is restricted to:
1. **Original author** (copyright holder)
2. **Authorized personnel** under explicit written agreement
3. **Acquiring entity** after successful transaction closing

---

## For Authorized Personnel

If you have been granted authorized access to this repository:

### Prerequisites
- Signed confidentiality/NDA agreement
- Written authorization from copyright holder
- Clear scope of authorized activities

### Development Guidelines

**1. Code Quality Standards**
- Follow existing code structure and style
- Maintain 90%+ test coverage
- Use Black for formatting (`black grcm tests`)
- Use isort for imports (`isort grcm tests`)
- Run full test suite before commits (`pytest`)

**2. Testing Requirements**
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=grcm --cov-report=html

# Run specific test types
pytest -m unit          # Unit tests
pytest -m integration   # Integration tests
pytest -m stress        # Stress tests
```

**3. Code Style**
- **Line length**: 100 characters
- **Quotes**: Double quotes for strings
- **Docstrings**: Google style
- **Type hints**: Recommended but optional

**4. Commit Messages**

Follow Conventional Commits format:
```
feat: add new qualia state detection
fix: resolve memory leak in attention module
docs: update API reference for phi calculation
test: add integration tests for desire alignment
```

**5. Confidentiality Requirements**

You MUST:
- Maintain strict confidentiality of all code and documentation
- Not share, copy, or distribute any materials
- Not discuss technical details outside authorized channels
- Not use this code outside authorized scope
- Report any security concerns immediately to copyright holder

You MUST NOT:
- Share code with unauthorized parties
- Commit to public repositories
- Create public forks or clones
- Discuss on public forums, social media, or technical communities
- Use trade secrets in other projects without authorization

---

## Development Setup (Authorized Personnel Only)

### 1. Clone Repository
```bash
# Clone from authorized private repository
git clone [AUTHORIZED_PRIVATE_REPO_URL]
cd GRCM
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
pip install -e ".[dev,all]"

# Verify installation
python -c "import grcm; print(grcm.__version__)"
```

### 4. Run Tests
```bash
pytest --cov=grcm
```

---

## Branching Strategy (Authorized Personnel Only)

If authorized to develop:

**Branch Naming**:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `test/` - Test additions/updates
- `perf/` - Performance improvements

**Workflow**:
```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes, commit
git add .
git commit -m "feat: description of changes"

# Push to private repository
git push origin feature/your-feature-name
```

**DO NOT**:
- Push to public repositories
- Create public branches or forks
- Share branch names or commit messages publicly

---

## Documentation (Authorized Personnel Only)

If authorized to update documentation:

**Update relevant files**:
- Code docstrings (Google style)
- Technical documentation in `docs/`
- Architecture diagrams if needed
- API reference materials

**Build documentation**:
```bash
cd docs
make html
```

**Confidentiality**: All documentation is confidential and proprietary.

---

## Acquisition Integration Support

After successful acquisition, the original author will provide:

**12-Month Integration Support**:
- Remote availability (20 hours/week)
- Email/Slack response within 24 hours
- Monthly video calls with engineering teams
- Assistance with deployment across portfolio companies
- Bug fixes for critical issues
- Optimization guidance

**Scope**:
- Integration assistance with existing systems
- Training for engineering teams
- Performance optimization
- Deployment troubleshooting
- Architecture guidance

---

## Contact for Authorized Inquiries

**For acquisition discussions**:
[Your Name]
[Your Email]
[Your Phone]

**For authorized development access**:
Must be pre-approved in writing by copyright holder.

**For security concerns**:
Report immediately to copyright holder via secure channel.

---

## Legal Notice

This repository and all contents are:
- **Proprietary and confidential**
- **Protected as trade secrets**
- **Subject to copyright protection**
- **Available for exclusive acquisition only**

Unauthorized access, use, copying, modification, or distribution may result in:
- Immediate termination of access
- Legal action for breach of confidentiality
- Claims for misappropriation of trade secrets
- Injunctive relief and damages

---

## Questions?

If you have been granted authorized access and have questions:
1. Review technical documentation in `docs/`
2. Contact copyright holder directly via authorized channels
3. Do NOT discuss on public forums or platforms

If you do NOT have authorized access:
- Contact copyright holder for acquisition inquiries only
- This is not available for public contribution, licensing, or use

---

**This is proprietary software. All rights reserved.**
