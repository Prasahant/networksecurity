from setuptools import find_packages, setup
from typing import List

def get_requirements()->List[str]:
    """
    this function willl return list of requirements
    
    A setup file contains all the necessary compressed data
    and instructions required to install a software 
    application on a computer. It automates the installation
    process by extractingfiles to the correct directories,
    configuring system settings, and creating shortcuts.
    
    Returns:
        List[str]: _description_
    """
    requirement_list:List[str] =[]
    
    try: 
        with open('requirements.txt','r') as file:
            lines = file.readlines() ## read lines from the files
            for line in lines: #3 process or travel through each line
                requirement = line.strip() # used to ignore empty lines and -e.
                if requirement and requirement!='-e.':
                    requirement_list.append(requirement)
     
    except FileNotFoundError:
        print("requirements.txt file not found")
    
    return requirement_list

# print(get_requirements())

setup(
    name="NetworkSecurity", 
    version = "0.0.1",
    author="Prashant Kumar Rajhans",
    author_email="prashantbabu980123@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)