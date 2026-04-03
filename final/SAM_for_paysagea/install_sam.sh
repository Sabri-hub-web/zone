#!/bin/bash
# Script d'installation rapide pour SAM vit_b

echo "=========================================="
echo "  Installation de SAM vit_b"
echo "=========================================="

# Couleurs pour le terminal
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Vérifier Python
echo -e "\n${YELLOW}[1/5]${NC} Vérification de Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓${NC} Python trouvé : $PYTHON_VERSION"
else
    echo -e "${RED}✗${NC} Python 3 n'est pas installé"
    exit 1
fi

# Cloner le repository
echo -e "\n${YELLOW}[2/5]${NC} Clonage du repository SAM..."
if [ -d "segment-anything" ]; then
    echo -e "${GREEN}✓${NC} Repository déjà cloné"
    cd segment-anything
else
    git clone https://github.com/facebookresearch/segment-anything.git
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} Repository cloné avec succès"
        cd segment-anything
    else
        echo -e "${RED}✗${NC} Erreur lors du clonage"
        exit 1
    fi
fi

# Installer SAM
echo -e "\n${YELLOW}[3/5]${NC} Installation de SAM..."
pip install -e . --quiet
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} SAM installé"
else
    echo -e "${RED}✗${NC} Erreur lors de l'installation de SAM"
    exit 1
fi

# Installer les dépendances
echo -e "\n${YELLOW}[4/5]${NC} Installation des dépendances..."
pip install opencv-python matplotlib pillow numpy --quiet
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Dépendances installées"
else
    echo -e "${RED}✗${NC} Erreur lors de l'installation des dépendances"
    exit 1
fi

# Télécharger le modèle vit_b
echo -e "\n${YELLOW}[5/5]${NC} Téléchargement du modèle vit_b (375 MB)..."
if [ -f "sam_vit_b_01ec64.pth" ]; then
    echo -e "${GREEN}✓${NC} Modèle déjà téléchargé"
else
    wget -q --show-progress https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} Modèle téléchargé"
    else
        echo -e "${RED}✗${NC} Erreur lors du téléchargement"
        echo "Vous pouvez le télécharger manuellement depuis :"
        echo "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        exit 1
    fi
fi

# Vérifier PyTorch
echo -e "\n${YELLOW}[BONUS]${NC} Vérification de PyTorch..."
python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA disponible:', torch.cuda.is_available())" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} PyTorch détecté"
else
    echo -e "${YELLOW}⚠${NC}  PyTorch non détecté - sera installé automatiquement"
fi

echo ""
echo "=========================================="
echo -e "${GREEN}✅ Installation terminée !${NC}"
echo "=========================================="
echo ""
echo "Pour tester SAM, utilisez :"
echo "  python test_sam.py votre_image.jpg"
echo ""
echo "Exemples de notebooks disponibles dans :"
echo "  segment-anything/notebooks/"
echo ""
