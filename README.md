# University Recommendation System

An intelligent university recommendation system built with Python that uses K-means clustering to match students with suitable universities based on their qualifications, preferences, and academic interests.

## Overview

This system analyzes student profiles (SAT scores, ACT scores, tuition preferences, admission rates, and academic major interests) and recommends the top 5 universities that best match their criteria. The application features a user-friendly GUI interface for easy interaction.

## Features

- **K-means Clustering**: Groups universities into 7 clusters based on their characteristics
- **Custom Distance Metrics**: Uses a hybrid approach combining Euclidean distance (for numerical features) and Hamming distance (for categorical features)
- **Interactive GUI**: User-friendly Tkinter interface to input preferences and view recommendations
- **Multiple Search Criteria**: Filter by SAT/ACT scores, tuition costs, admission rates, completion rates, region, gender preference, and academic majors
- **Comprehensive Major Support**: 39 different academic fields mapped to CIP (Classification of Instructional Programs) codes
- **Real-time Recommendations**: Returns up to 12 top matching universities instantly

## Prerequisites

Before running this application, ensure you have the following installed:

### Required Python Libraries

- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning tools (for MinMaxScaler)
- **matplotlib** - Data visualization
- **openpyxl** - Excel file handling
- **tkinter** - GUI framework (usually comes with Python)

### System Requirements

- Python 3.6 or higher
- At least 2GB of RAM

## Installation

1. Clone the repository:
```bash
git clone https://github.com/DuaAliAnsari/University-Recommendation-System.git
cd University-Recommendation-System
