/**
 * 
 */
package openservo;


/**
 * 
 * Clase para el acceso desde Java a las funcione de la librería OSIF
 * Al no ser public solo se puede invocar desde dentro del paquete openservo.
 * De esta manera será la clase openservo la única que interactuará con ella.
 * 
 * @author Alberto Hamilton
 *
 */
class OsifJNI {
	/**
	 * Initialise the OSIF USB interface. Enumerates all connected OSIF devices.
	 * @return <0 error 1 success
	 **/
	public static native int OSIF_init();

	/**
	 *De-Initialise the OSIF USB interface
	 **/
	public static native int OSIF_deinit();

	/**
	 * Return the OSIF library version for compatibility checks
	 * @return an integer of the version number in  majorminor xxyy
	 **/
	public static native int OSIF_get_libversion();

	/**
	 *Write data to the I2C device.
	 *This will start an I2C transaction (with automatic restart)
	 *and write buflen bytes to address addr
	 *
	 *This assumes the device needs a register selection before
	 *doing the write. Some devices don't require this, and
	 *you should either use OSIF_writeonly or put the first byte
	 *to write in the addr register and start writing from +1
	 *offset in data.
	 *
	 *@param adapter integer of the adapter scanned. 0 indexed.
	 *@param i2c_addr integer address of the device.
	 *@param addr the register address in the device to read
	 *@param data passed in buffer to be written
	 *@param buflen number of bytes to read. Current hardware limit is 64 bytes
	 *@param issue_stop issue the stop bit at the end of the transaction?
	 *
	 *@return <0 error 1 success
	 **/
	public static native int OSIF_write_data(int adapter, int i2c_addr, byte addr, byte[] data, int buflen, boolean issue_stop );


	/** Shortcut to the above function. This will always send a stop at the end of the write **/
	public static int OSIF_write(int adapter, int i2c_addr, byte addr, byte[] data , int buflen) {
		return OSIF_write_data(adapter, i2c_addr, addr, data,buflen, true );
	}

	/**
	 *Write data to the I2C device.
	 *This will start an I2C transaction (with automatic restart)
	 *and write buflen bytes to address addr
	 *
	 *This assumes the device does NOT need a register selection before
	 *doing the write. Some devices do require this, and
	 *you should either use OSIF_write or put the register selection
	 *byte at element 0 in your data string
	 *
	 *@param adapter integer of the adapter scanned. 0 indexed.
	 *@param i2c_addr integer address of the device.
	 *@param data passed in buffer to be filled
	 *@param buflen number of bytes to read. Current hardware limit is 64 bytes
	 *@param issue_stop do we want to send the I2C stop signal after the
	 *			transaction? issue_stop will switch off the request
	 *
	 *@return <0 error 1 success
	 **/
	public static native int OSIF_writeonly(int adapter, int i2c_addr, byte[] data, int buflen, boolean issue_stop );

	/**
	 *Read from the I2C device at address addr
	 *will fill data into the read buffer.
	 *
	 *Note:
	 *This function will do a write before a read
	 *with a restart.
	 *
	 *@param adapter integer of the adapter scanned. 0 indexed.
	 *@param i2c_addr integer address of the device.
	 *@param addr the register address in the device to read
	 *@param data passed in buffer to be filled
	 *@param buflen number of bytes to read. read in small chunks. 64 bytes is a realistic figure
	 *@param issue_stop issue the stop bit at the end of the transaction?
	 *
	 *@return <0 error 1 success
	 **/
	public static native int OSIF_read_data(int adapter, int i2c_addr, byte addr, byte[] data, int buflen, boolean issue_stop );

	/** Shortcut to the above function with an I2C stop bit **/
	public static int OSIF_read(int adapter, int i2c_addr, byte addr, byte[] data, int buflen)
	{
		return OSIF_read_data(adapter, i2c_addr, addr, data, buflen, true);
	}

	/**
	 *Read from the I2C device at address addr
	 *will fill data into the read buffer.
	 *
	 *Note:
	 *This function will NOT do a write before a read
	 *it will only perform a read. Make sure the I2C
	 *device is setup for this read only transfer
	 *by using OSIF_write, or alternatively be
	 *sure your device supports this method of communication
	 *
	 *@param adapter integer of the adapter scanned. 0 indexed.
	 *@param i2c_addr integer address of the device.
	 *@param data passed in buffer to be filled
	 *@param issue_stop issue the stop bit at the end of the transaction?
	 *
	 *@return <0 error 1 success
	 **/
	public static native int OSIF_readonly(int adapter, int i2c_addr, byte[] data, int buflen, boolean issue_stop );


	/**
	 *Scan the I2C bus for devices.
	 *
	 *Scan the I2C bus by addressing all devices (0x01 to 0x7F) in turn and waiting
	 *to see if we get an ACK
	 *Note not all devices work like this, and can send some devices into an unknown
	 *state. BE CAREFUL.
	 *
	 *@param adapter integer of the adapter scanned. 0 indexed.
	 *
	 *@return NULL si error, ó array de dispositivos encontrados
	 **/
	public static native int[] OSIF_scan(int adapter);

	/**
	 *Probe a device at a given address to see if it will ACK
	 *
	 *@param adapter integer of the adapter scanned. 0 indexed.
	 *@param i2c_addr integer address of the device.
	 *
	 *@return true if a device is found at address
	 **/
	public static native boolean OSIF_probe(int adapter, int i2c_addr );

	/**
	 *Write 1 to a register in the device in one transaction.
	 *generally used for "command" functions in I2C slave
	 *devices that will trigger a function from a write to
	 *a register.
	 *
	 *@param adapter integer of the adapter scanned. 0 indexed.
	 *@param i2c_addr integer address of the device.
	 *@param command the register to write to.
	 *
	 *@return <0 error 1 success
	 **/
	public static native int OSIF_command(int adapter, int i2c_addr, byte command);
	
	/**
	 *Get a count of the connected OSIF adapters:
	 *
	 *@return number of connected OSIF adapters
	 **/
	public static native int OSIF_get_adapter_count();

	/**
	 *Query the connected OSIF for its name.
	 *May also be used for firmware version.
	 *
	 *@param adapter integer of the adapter scanned. 0 indexed.
	 *
	 *@return character string filled with the name
	 **/
	public static native String OSIF_get_adapter_name(int adapter);
	
	
	/**
	 *GPIO Control function to set the direction of the pins, and is enabled
	 *    If you set the I2C pins (SDA SCL) as outputs, you will disable the
	 *    i2c module. Be warned!
	 *@param adapter_no integer of the adapter scanned. 0 indexed.
	 *@param ddr a bitwise OR of the pins to set as input(0) or output(1)
	 *    eg gpio1(TX) and gpio2(RX)  00000011 or 0x03
	 *@param enabled a bitwise OR to enable the pin (1) for future writes. Can be used as a mask
	 * TX line
	 * RX line
	 * MISO
	 * MOSI
	 * SDA
	 * SCL
	 **/
	public static native int OSIF_io_set_ddr(int adapter_no, int ddr, int enabled);

	/**
	 *GPIO Control function to set pin high or low
	 *    (only works if ddr set to output)
	 *
	 *@param adapter_no integer of the adapter scanned. 0 indexed.
	 *@param io a bitwise OR of the pins to set high (1) or low (0)
	 *    eg gpio1(TX) and gpio2(RX)  00000011 or 0x03 to set those high
	 **/
	public static native int OSIF_io_set_out(int adapter_no, int io);
	
	/**
	 *GPIO Control function to read the status of all gpio pins
	 *
	 *@param adapter integer of the adapter scanned. 0 indexed.
	 **/
	public static native int OSIF_io_get_in(int adapter);

	/**
	 *Shortcut functions to update one pin only
	 **/
	public static native int OSIF_io_set_out1(int adapter_no, int gpio, int state);

	/**
	 *get the current pin states. THIS IS NOT suitable for asking the osif what pins
	 *are set when this lib loaded, OSIF doesn't know that. It is this lib that keeps
	 *tabs on what pins are being set/uset
	 *@param adapter integer of the adapter scanned. 0 indexed.
	 **/
	public static native int OSIF_io_get_current(int adapter);

	/**
	 *Disable the I2C port in case we want to use the pins as gpios,
	 *or if we want to reset the I2C module
	 *@param adapter integer of the adapter scanned. 0 indexed.
	 **/
	public static native int OSIF_disable_i2c(int adapter);
	
	/**
	 *Enable a previously disabled I2C module
	 **/
	public static native int OSIF_enable_i2c(int adapter);
	
	
	public static native int OSIF_set_bitrate(int adapter_no, int bitrate_hz);
	public static native int OSIF_set_twbr(int adapter_no, int twbr, int twps);

	
	static {
		System.out.println("java.library.path:"
				+System.getProperty("java.library.path"));
       //System.loadLibrary( "openservo_OsifJNI.so" );
		String libreria=System.getProperty("user.dir")+"/lib/openservo/openservo_OsifJNI.so";
		System.out.println("Tratamos de cargar: "+libreria);
	   System.load(libreria);
    }

	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		System.out.println("\n Comienza el programa");
		
		int intentos=4;
		int numAdap=0;
		while(true) {
			if(intentos==0) {
				System.out.println("\nPasaron los intentos\n");
				System.exit(0);
			}

			System.out.println("\n Inicializamos OSIF");
			if(OSIF_init()<0) {
				System.out.println("\nProbema al inicializar OSIF\n");
				System.exit(1);
			}

			System.out.println("\n La versión de la librería OSIF es "+OSIF_get_libversion());

			numAdap=OSIF_get_adapter_count();
			System.out.println("\n Obtenemos numero de adaptadores OSIF:"+numAdap);
			if((numAdap)>0) {
				break;
			}
			intentos--;
			OSIF_deinit();
			try { Thread.sleep(3000); } catch (Exception e) {}
		}
		System.out.println("\n Obtenemos nombre de los adaptadores OSIF");
		for(int adp=0; adp<numAdap; adp++) {
			String nomAdap;

			if((nomAdap=OSIF_get_adapter_name(adp))==null) {
				System.out.println("\nAdaptador "+adp+"No tiene nombre");
			} else {
				System.out.println("\nNombre del Adaptador "+adp+": >"+nomAdap+"<");
			}
			
			int dispos[];
			if((dispos=OSIF_scan(adp))==null) {
				System.out.println("\nProblema al obtener dispositivoe en adaptador "+adp);
				continue; //pasamos al siguiente adaptador
			}
			System.out.println("\nAdaptador "+adp+" tiene "+dispos.length+" dispositivos I2C");
			
			for(int ida=0; ida<dispos.length; ida++) {
				int da=dispos[ida];
				if(!OSIF_probe(adp, da )) {
					System.out.println("\nEl dispositivo "+da+" NO respondió");
					continue; //pasamos al siguiente dispositivo
				}
				System.out.println("\nEl dispositivo "+da+" respondió");
				
				//Vemos si es OpenServo
				byte[] buffer=new byte[4];
				if(OSIF_read(adp, da, (byte) 0x00, buffer, 1)<0) {
					System.out.println("\n Error al leer de dispositivo "+da);
				} else
					if(buffer[0]==0x01) {
						System.out.println("\nEl dispositivo "+da+" ES OpenServo");
					} else {
						System.out.println("\nEl dispositivo "+da+" NO es OpenServo");
						continue; //siguiente dispositivo
					}
				
				//Obtenemos más datos
				if(OSIF_read(adp, da, (byte) 0x00, buffer, 4)<0) {
					System.out.println("\n Error al leer de dispositivo "+da);
				} else
					System.out.println(String.format("\n\tTipo %d.%d  Version: %d.%d"
							,buffer[0],buffer[1],buffer[2],buffer[3]));
				
				for(int veces=10; veces>0; veces--) {
					//Timer
					if(OSIF_read(adp, da, (byte) 0x06, buffer, 2)<0){
						System.out.println("\n Error al leer de dispositivo "+da);
					} else {
//						//Volcamos en hexadecimal
//						for (int i=0; i<buffer.length; i++)
//						System.out.printf(" %02X",buffer[i]);
//						System.out.printf("\n");


						int vel=((((int)buffer[0])&0xff)<<8) + (((int)buffer[1])&0xff);
						System.out.println("\tTimer:"+(vel));
					}
					//posicion
					if(OSIF_read(adp, da, (byte) 0x08, buffer, 2)<0){
						System.out.println("Error al leer de dispositivo "+da);
					} else {

						int pos=((((int)buffer[0])&0xff)<<8) + (((int)buffer[1])&0xff);
						System.out.println("\tPosicion:"+pos);
					}
					//velocidad
					if(OSIF_read(adp, da, (byte) 0x0A, buffer, 2)<0){
						System.out.println("Error al leer de dispositivo "+da);
					} else {
						int vel=((((int)buffer[0])&0xff)<<8) + (((int)buffer[1])&0xff);
						System.out.println("\tVelocidad:"+vel);
					}
					//Timer
					if(OSIF_read(adp, da, (byte) 0x06, buffer, 2)<0){
						System.out.println("Error al leer de dispositivo "+da);
					} else {
						int vel=((((int)buffer[0])&0xff)<<8) + (((int)buffer[1])&0xff);
						System.out.println("\tTimer:"+vel);
					}
					try { Thread.sleep(3000); } catch (Exception e) {}					
				}
			}

		}
		
		System.out.println("\nCerramos la conexion a OSIF");
		OSIF_deinit();
	}
}
