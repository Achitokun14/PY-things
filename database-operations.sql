-- Initialize Database
CREATE DATABASE tp_1;
USE tp_1;

-- 1. Create table salle (room)
CREATE TABLE salle (
    numero INT,
    nom VARCHAR(30),
    capacité INT
);

-- 2. Add primary key constraint to room number
ALTER TABLE salle 
ADD CONSTRAINT pk_salle_num PRIMARY KEY(numero);

-- 3. Set default capacity to 30
ALTER TABLE salle 
MODIFY capacité INT DEFAULT(30);

-- 4. Make room name mandatory
ALTER TABLE salle 
MODIFY nom VARCHAR(30) NOT NULL;

-- 5. Create computer table with required fields
CREATE TABLE ordinateur (
    référence VARCHAR(30) PRIMARY KEY,
    marque VARCHAR(30) DEFAULT('HP'),
    model VARCHAR(30),
    prix FLOAT
);

-- 6. Add price constraint between 2000 and 15000
ALTER TABLE ordinateur 
ADD CONSTRAINT prix_check CHECK(prix BETWEEN 2000 AND 15000);

-- 7. Add room number column to computer table
ALTER TABLE ordinateur 
ADD COLUMN num_salle INT;

-- 8. Add foreign key constraint with cascade delete and null update
ALTER TABLE ordinateur 
ADD CONSTRAINT foreign_numsalle_ord 
FOREIGN KEY(num_salle) REFERENCES salle(numero)
ON DELETE CASCADE 
ON UPDATE SET NULL;

-- 9. Modify price constraint to >= 1800
ALTER TABLE ordinateur 
DROP CONSTRAINT prix_check;
ALTER TABLE ordinateur 
ADD CONSTRAINT prix_check CHECK(prix >= 1800);

-- 10. Allow null values for room name
ALTER TABLE salle 
MODIFY nom VARCHAR(30) NULL;

-- 11. Change default capacity to 25
ALTER TABLE salle 
MODIFY capacité INT DEFAULT(25);

-- 12. Insert room data
INSERT INTO salle (nom, numero, capacité) VALUES 
('khawarismi', 1, 20),
('victore hugo', 2, DEFAULT);

-- 13. Insert computer data
INSERT INTO ordinateur (référence, marque, model, prix, num_salle) VALUES
('O1', 'Revillon', NULL, 5600, 1),
('02', 'DELL', 'Latitude', 7000, 2);

-- 14. Update room number and verify
UPDATE salle SET numero = 11 WHERE numero = 1;
SELECT * FROM ordinateur WHERE num_salle IS NULL;

-- 15. Delete room 2 and verify computer deletion
DELETE FROM salle WHERE numero = 2;
SELECT * FROM ordinateur WHERE référence = '02';

-- 16. Update Dell Latitude prices
UPDATE ordinateur 
SET prix = 9000 
WHERE marque = 'DELL' AND model = 'Latitude';

-- 17. Delete rooms with capacity < 10
DELETE FROM salle WHERE capacité < 10;

-- 18. Update computer O1 details
UPDATE ordinateur 
SET marque = 'toshibat', 
    model = 'protege', 
    prix = 6200 
WHERE référence = 'O1';
