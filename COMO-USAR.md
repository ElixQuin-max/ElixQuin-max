# Cómo montar tu perfil

1. **Crea el repo especial**: en GitHub crea un repositorio público con el **mismo nombre que tu usuario** (ej. si eres `eli-dev`, el repo es `eli-dev/eli-dev`). Lo que esté en su `README.md` aparece en tu perfil.
2. **Edita `profile.config.json`**: cambia `github_username` (se propaga a todo), tu LinkedIn en `socials`, y ajusta stack, radares y textos a tu gusto.
3. **Tu foto (opcional pero recomendado)**: guarda `assets/source/portrait.png`, de hombros hacia arriba y con el fondo quitado (remove.bg o similar). Sin foto, el banner usa "ELI".
4. **Genera**:
   ```bash
   pip install -r scripts/requirements.txt
   python scripts/generate.py
   ```
5. **Sube todo** (`git add . && git commit -m "perfil" && git push`). Desde ahí, el workflow de `.github/workflows/profile.yml` regenera el banner y el README cada vez que cambies el config o la foto (actívalo en Settings → Actions → General → *Read and write permissions*).

## Personalizar

- **Logos del banner**: `banner_logos` usa los PNG de `scripts/logos/`. Para añadir otro, pon un PNG con fondo transparente (ej. `nextdotjs.png`) y agrega su nombre a la lista. Más logos = SVG más pesado (cada uno suma ~150 KB).
- **Iconos de la tabla**: `tech_stack[].icons` usa los ids de [skillicons.dev](https://skillicons.dev).
- **Colores**: paletas en `THEMES` (banner) y `RADAR` dentro de `scripts/generate.py`.
- **No edites `README.md` a mano**: se sobrescribe al generar. Cambia el config o la función `render_readme()`.

Diseño inspirado en el perfil de [@macu-dev](https://github.com/macu-dev).
