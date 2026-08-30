from setuptools import setup, find_packages

packages = [
    "MAPS",
    "MAPS.integration_P",
    "MAPS.integration_P.datasets",
    "MAPS.integration_P.model",
    "MAPS.integration_P.utils",
]

package_dir = {
    "MAPS": "MAPS",
    "MAPS.integration_P": "MAPS/integration_P",
    "MAPS.integration_P.datasets": "MAPS/integration_P/datasets",
    "MAPS.integration_P.model": "MAPS/integration_P/model",
    "MAPS.integration_P.utils": "MAPS/integration_P/utils",
}

setup(
    name='MAPS',
    version='0.1.2',
    url='https://github.com/dbjzs/MAPS',
    author='Bingjie Dai',
    author_email='17516970902@163.com',
    license='Apache',
    packages=packages,
    package_dir=package_dir,
    package_data={
        "MAPS.integration_P.datasets": ["gene.csv"],
    },
    description="Unifying physical and molecular coordinates systems across modalities in spatial biology",
    zip_safe=False
)


